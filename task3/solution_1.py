import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

domain = np.array([[0, 5]])

np.random.seed(2)


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.kernel_f = 0.5 * Matern(length_scale=0.5, nu=2.5)
        self.kernel_v = pow(2, 0.5) * Matern(length_scale=0.5, nu=2.5)
        self.pri_mean_v = 1.5
        self.train_x = []
        self.train_f = []
        self.train_v = []
        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=0.15)
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=0.0001)
        self.ucb_beta = 1.5
    

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        res = self.optimize_acquisition_function()
        return np.atleast_2d(res)

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        # mean_f, std_f = self.gp_f.predict(x.reshape(-1, 1), return_std=True)
        # mean_v, std_v = self.gp_v.predict(x.reshape(-1, 1), return_std=True)

        mean_f, std_f = self.gp_f.predict(np.atleast_2d(x), return_std=True)
        mean_v, std_v = self.gp_v.predict(np.atleast_2d(x), return_std=True)

        prob_constraint = norm.cdf(1.2, loc=mean_v[0]+self.pri_mean_v, scale=std_v[0])
	
        penalty_coef = 3

        method = 'EI'

        if (method == 'UCB'):
            ucb = mean_f[0] + self.ucb_beta * std_f[0]
            
            # return ucb.flatten()
            # if self.pri_mean_v + mean_v[0] - 0.2 * std_v[0] <= 1.2:
            #     return ucb - penalty
            # else:
            #     return ucb

        if (method == 'EI'):
            mean_sample = []
            for i in range(len(self.train_x)):
                mean_sample.append(self.gp_f.predict(np.array(self.train_x[i]).reshape(-1, 1))[0])

            mu_sample_opt = np.max(mean_sample)

            xi = 0.01

            imp = mean_f[0] - mu_sample_opt - xi
            
            Z = imp / std_f

            # EI = imp * norm.cdf(Z) + std_f * norm.pdf(Z)
            
            EI = std_f * (Z * norm.cdf(Z) + norm.pdf(Z))
            return EI +(1- prob_constraint) * 1.5

            # if self.pri_mean_v + mean_v - 0.2 * std_v <= 1.2:
            #     return (EI - penalty).ravel()
            # else:
            #    return EI.ravel()


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.train_x.append(x)
        self.train_f.append(f)
        self.train_v.append(v)
        # self.train_f = np.array(self.train_f)
        # self.train_x = self.train_x.reshape(-1, 1)
        # self.train_f = self.train_f.reshape(-1, 1)
        self.gp_f.fit(self.train_x, self.train_f)
        self.gp_v.fit(self.train_x, self.train_v)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        
        max_f = -1e10
        index = -1
        for i in range(len(self.train_x)):
            if self.train_v[i] > 1.2 and self.train_f[i] > max_f:
                max_f = self.train_f[i]
                index = i
        return self.train_x[index]
        
        # return self.train_x[self.train_f.index(max(self.train_f))]


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()
    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
            1, n_dim)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)
    
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
