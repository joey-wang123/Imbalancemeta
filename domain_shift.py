
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from scipy.stats import norm
from scipy import optimize as opt

class Online_domainshift_detection():
    """
    Initialize an estimator.

    Parameters
    ----------
    thresholding_method: str,
        'adapt': adaptive threshold, estimate online the mean and variance of the statistics,
        then take the appropriate quantile under the assumption that it is approximately gaussian;
        'fixed': fixed to some value.
    thresholding_quantile: float,
        if adaptive threshold, quantile for online estimate of mean and variance of the statistics.
    fixed_threshold: float,
        if fixed threshold, value of fixed threshold
    adapt_forget_factor: float,
        forgetting factor for online estimation of threshold
    store_values: bool
        if True, store all the values of the statistic, threshold, detection result
        if False, does not store anything, and self.update(sample) returns the detection result directly

    Attributes
    ----------
    statistic: float,
        current value of the detection statistic.
    adapt_mean: float,
        current estimate for the mean of the squared statistic.
    adapt_second_moment: float,
        current estimate for the 2nd order moment of the squared statistic.
    stat_stored: list,
        history of the statistic.

    """
    def __init__(self,
                 init_sample,
                 kernel_func=lambda X, Y: np.dot(Y, np.array(X).T),
                 window_size=100, nbr_windows=5,
                 thresholding_method='adapt',
                 thresholding_quantile=0.95,
                 fixed_threshold=None,
                 adapt_forget_factor=0.05,
                 store_values=True):
        super().__init__()

        self.kernel_func = kernel_func
        self.B = window_size
        self.N = nbr_windows
        # what do we store
        kxx = kernel_func(init_sample[np.newaxis, :], init_sample)
        self.kernel_sum_XX = kxx * np.ones(self.N)
        self.kernel_sum_XY = kxx * np.ones(self.N)
        self.kernel_sum_YY = kxx
        self.X = []
        self.adapt_forget_factor = adapt_forget_factor
        self.thresholding_quantile = thresholding_quantile
        self.thresholding_method = thresholding_method
        self.thresholding_mult = norm.ppf(thresholding_quantile)

        self.adapt_mean = 0
        self.adapt_second_moment = 0

        self.store_values = store_values
        self.stat_stored = []

        print('self.adapt_forget_factor', self.adapt_forget_factor)
        print('thresholding_quantile', thresholding_quantile)
        for i in range(self.N):
            temp = deque()
            for j in range(self.B):
                temp.append(init_sample)
            self.X.append(temp)
        temp = deque()
        for j in range(self.B):
            temp.append(init_sample)
        self.Y = temp

    def apply_to_data(self, data):
        """Apply the algorithm to an entire collection of data.

        Parameters
        ----------
        data: np.ndarray (n,d),
            array of n samples.

        """
        for d in data:
            self.update(d)

    def flag_sample(self):
        """After computation of statistic, flag sample according to chosen thresholding method.

        Returns
        -------
        r: bool,
            detection result.
        """
        if self.thresholding_method == 'adapt':
            return self.statistic > np.sqrt(
                self.adapt_mean + self.thresholding_mult * np.sqrt(self.adapt_second_moment - self.adapt_mean ** 2))
        elif self.thresholding_method == 'fixed':
            return self.statistic > self.fixed_threshold
        else:
            return TypeError('Thresholding method not recognised.')

    def update_kernel_sum(self, val, datax, datay, newx, newy, oldx, oldy):
        return val + (self.kernel_func(datax, newy).sum() - self.kernel_func(datax, oldy).sum()
                      + self.kernel_func(datay, newx).sum() - self.kernel_func(datay, oldx).sum()
                      + self.kernel_func(newx[np.newaxis, :], newy) - self.kernel_func(oldx[np.newaxis, :], oldy)) / (
            (len(datax) + 1) * (len(datay) + 1))

    def update(self, new_sample):
        """Process the arrival of a new sample. If store_value is True, store detection statistic,
        threshold and detection result.

        Parameters
        ----------
        new_sample: np.ndarray (d, ),
            new sample.

        Returns
        -------
        res: bool,
            detection result
        """
        self.statistic = self.update_stat(
            new_sample)  # compute the new detection statistic b the user-implemented function

        # compute adaptive detection result
        #print('self.adapt_forget_factor', self.adapt_forget_factor)
        self.adapt_mean = (
            1 - self.adapt_forget_factor) * self.adapt_mean + self.adapt_forget_factor * self.statistic ** 2
        self.adapt_second_moment = (
            1 - self.adapt_forget_factor) * self.adapt_second_moment + self.adapt_forget_factor * self.statistic ** 4

        res = self.flag_sample()
        # if history is stored
        if self.store_values:
            thres = np.sqrt(
                self.adapt_mean + self.thresholding_mult * np.sqrt(self.adapt_second_moment - self.adapt_mean ** 2))
            self.stat_stored.append((self.statistic, thres, res))

        return res  # return the result


    def update_stat(self, sample):
        sample_y = self.Y.popleft()
        sample_x = []
        for i in range(self.N):
            sample_x.append(self.X[i].popleft())
        # add sample to Y
        self.kernel_sum_YY = self.update_kernel_sum(self.kernel_sum_YY,
                                                    self.Y, self.Y,
                                                    sample, sample,
                                                    sample_y, sample_y)
        for i in range(self.N - 1):
            self.kernel_sum_XX[i] = self.update_kernel_sum(self.kernel_sum_XX[i],
                                                           self.X[i], self.X[i],
                                                           sample_x[i + 1], sample_x[i + 1],
                                                           sample_x[i], sample_x[i])
            self.kernel_sum_XY[i] = self.update_kernel_sum(self.kernel_sum_XY[i],
                                                           self.X[i], self.Y,
                                                           sample_x[i + 1], sample,
                                                           sample_x[i], sample_y)
        self.kernel_sum_XX[-1] = self.update_kernel_sum(self.kernel_sum_XX[-1],
                                                        self.X[-1], self.X[-1],
                                                        sample_y, sample_y,
                                                        sample_x[-1], sample_x[-1])
        self.kernel_sum_XY[-1] = self.update_kernel_sum(self.kernel_sum_XY[-1],
                                                        self.X[-1], self.Y,
                                                        sample_y, sample,
                                                        sample_x[-1], sample_y)

        # roll out old data

        for i in range(self.N - 1):
            self.X[i].append(sample_x[i + 1])
        self.X[-1].append(sample_y)
        self.Y.append(sample)

        return self.kernel_sum_XX.sum() / self.N + self.kernel_sum_YY - 2 * self.kernel_sum_XY.sum() / self.N



# %% utils specific to newma
def convert_parameters(window_size, forget_factor):
    """From the window_size and one forgetting factor, compute the other forgetting factor..
    """
    w_ = window_size
    C = forget_factor * (1 - forget_factor) ** w_

    # educated guess for initialization
    if forget_factor > 1 / (w_ + 1):
        init = 1 / (2 * (w_ + 1))
    else:
        init = 2 / (w_ + 1)

    def func(x):
        return (x * (1 - x) ** w_ - C) ** 2

    def grad(x):
        return ((1 - x) ** w_ - w_ * x * (1 - x) ** (w_ - 1)) * 2 * (x * (1 - x) ** w_ - C)

    return opt.minimize(func, jac=grad, x0=init, bounds=((0, 1),), tol=1e-20).x[0]


def select_optimal_parameters(window_size, grid_size=1000):
    """From the window_size, give the best newma parameters, w.r.t. the error bound in the paper.
    """
    def error_bound(L, l):
        numerator = (np.sqrt(L + l) + ((1 - l) ** (2 * window_size) - (1 - L) ** (2 * window_size)))
        denominator = ((1 - l) ** window_size - (1 - L) ** window_size)
        return numerator / denominator

    ax = np.exp(np.linspace(np.log(1.001 / (window_size + 1)), -0.01, grid_size))
    errors = np.zeros(grid_size)
    for ind, L in zip(range(grid_size), ax):
        l = convert_parameters(window_size, L)
        errors[ind] = error_bound(L, l)
    Lambda = (ax[np.argmin(errors)] + 1 / (window_size + 1)) / 2
    return Lambda, convert_parameters(window_size, Lambda)





    


