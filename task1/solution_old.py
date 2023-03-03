import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm

# more import
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        self.k_h = 0.05
        self.pri_std = 1
        self.pri_mean = 0
        # self.kernel = Matern(nu=1.5)
        # self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9, normalize_y=True, random_state=10)

    def k(self, x1, x2):
        """
        kernel function, RBF kernel
        """
        # parameter to be tune
        k_gaussian_1 = np.exp(-np.linalg.norm(x1-x2)**2/((0.5*self.k_h)**2))
        k_gaussian_2 = np.exp(-np.linalg.norm(x1 - x2) ** 2 / ((2*self.k_h) ** 2))

        return k_gaussian_1 + 0.2 * k_gaussian_2


    def make_predictions(self,test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(test_features.shape[0], dtype=float)
        gp_std = np.zeros(test_features.shape[0], dtype=float)

        N_test = test_features.shape[0]

        self.pri_mean_vector_test = np.zeros(N_test)
        for i in range(N_test):
            self.pri_mean_vector_test[i] = self.pri_mean

        # compute K_XM
        print(self.num_sample)
        K_XM = np.zeros((N_test, self.num_sample))
        for i in range(N_test):
            test_data = test_features[i, :]
            for j in range(self.num_sample):
                K_XM[i, j] = self.k(test_data, self.x_M[j, :])
        K_XM_T = np.transpose(K_XM)

        # the part to be inversed
        k_inner_inv = np.linalg.inv(np.matmul(self.K_MA, self.K_AM) + (self.pri_std ** 2) * self.K_MM)

        # pos_mean = pri_mean + K_XM* (K_MA * K_AM + pri_std^2 * K_MM)^(-1) * K_MA * (y_A - pri_mean)
        gp_mean = self.pri_mean_vector_test + np.matmul(np.matmul(K_XM, np.matmul(k_inner_inv, self.K_MA)), (self.y_A - self.pri_mean_vector))

        # compute stddev
        # for Nystrom method, subset of datapoints:
        # variance is computed by
        # pri_std^2 * K_XM.T * (K_MA * K_AM + pri_std^2 * K_MM)^(-1) * K_XM + k(x_star, x_star) - K_XM.T * K_MM^(-1) * K_XM
        for i in range(N_test):
            std_x_i = (self.pri_std ** 2) * np.matmul(K_XM[i, :], np.matmul(k_inner_inv, K_XM_T[:, i])) + self.k(test_features[i], test_features[i]) - np.matmul(K_XM[i, :], np.matmul(self.K_MM_inv, K_XM_T[:, i]))
            gp_std[i] = std_x_i

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here

        self.N = train_GT.shape[0]
        self.y_A = train_GT
        self.x_A = train_features
        self.pri_mean_vector = np.zeros(self.N)
        self.K_AA = np.zeros((self.N, self.N))
        self.I_AA = np.eye(self.N)

        # success when 3000
        self.num_sample = 200

        # Kmeans to determine subset of datapoints
        kmCluster = KMeans(n_clusters=self.num_sample)
        kmCluster.fit(self.x_A)
        self.x_M= kmCluster.cluster_centers_

        # M : datapoint set sampled from the whole dataset
        self.K_MM = np.zeros((self.num_sample, self.num_sample))
        self.K_MA = np.zeros((self.num_sample, self.N))


        # compute mean_vector_prior
        for i in range(self.N):
            self.pri_mean_vector[i] = self.pri_mean


        # compute K_MM
        for i in range(self.num_sample):
            for j in range(self.num_sample):
                self.K_MM[i, j] = self.k(self.x_M[i, :], self.x_M[j, :])

        self.K_MM_inv = np.linalg.inv(self.K_MM)

        # compute K_MA
        for i in range(self.num_sample):
            for j in range(self.N):
                self.K_MA[i, j] = self.k(self.x_M[i, :], self.x_A[j, :])

        self.K_AM = np.transpose(self.K_MA)

    def cross_validation(self, train_gt, train_x, numFolds):
        """
        k-fold cross validation to tune parameters
        can also use library function instead
        """

        print("Starting cross-validation...")

        X = train_x
        Y = train_gt

        folds = KFold(n_splits=numFolds, shuffle=True)

        estimators = []
        results = np.zeros(len(X))
        score = 0.0
        n_fold = 1

        for train_index, test_index in folds.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]

            model_cv = Model()
            model_cv.fitting_model(Y_train, X_train)

            prediction_cv = model_cv.make_predictions(X_test)

            score_fold = cost_function(Y_test, prediction_cv[0])
            score += score_fold

            print("Fold n°", n_fold, " Loss: ", score_fold)
            n_fold += 1

        score /= numFolds
        print("The averaged cross-validation loss is: ", score)



def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT, train_features)


    # # uncomment here when tuning parameters, can also commet the last line to not fit a whole model
    model.cross_validation(train_GT, train_features, 5)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
