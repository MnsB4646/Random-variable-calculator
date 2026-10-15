import numpy as np
from scipy.integrate import quad
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import IntegrationWarning

class distribution:
    def __init__(self, name, parameters, func, lower_limit, upper_limit, discrete=True, plot_lower=None, plot_upper=None):
        self.name = name
        self.parameters = parameters
        self.discrete = discrete
        self.func = func 
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.plot_lower = plot_lower if plot_lower is not None else lower_limit
        self.plot_upper = plot_upper if plot_upper is not None else upper_limit

    def get_support(self, truncation_limit=1000):
        if not self.discrete:
            return None
        low = int(self.lower_limit) if self.lower_limit != -np.inf else -truncation_limit
        high = int(self.upper_limit) if self.upper_limit != np.inf else truncation_limit
        return np.arange(low, high + 1)

    def __add__(self, other):
        return add_distributions(self, other)

    def __sub__(self, other):
        return subtract_distributions(self, other)

    def __mul__(self, other):
        return multiply_distributions(self, other)

    def __str__(self):
        return f"{self.name} with parameters {self.parameters}"


def add_distributions(x, y):
    new_lower = x.lower_limit + y.lower_limit
    new_upper = x.upper_limit + y.upper_limit
    new_plot_lower = x.plot_lower + y.plot_lower
    new_plot_upper = x.plot_upper + y.plot_upper
    new_discrete = x.discrete and y.discrete

    def evaluate_continuous(z):
        if z < new_lower or z > new_upper: return 0.0
        lower_bound = max(x.lower_limit, z - y.upper_limit)
        upper_bound = min(x.upper_limit, z - y.lower_limit)
        if lower_bound > upper_bound: return 0.0
        res, _ = quad(lambda t: x.func(t) * y.func(z - t), lower_bound, upper_bound, limit=100)
        return res

    def evaluate_discrete(z):
        if z < new_lower or z > new_upper or not float(z).is_integer(): return 0.0
        prob_sum = 0.0
        for k in x.get_support():
            if y.lower_limit <= (z - k) <= y.upper_limit:
                prob_sum += x.func(k) * y.func(z - k)
        return prob_sum

    cont_dist = x if not x.discrete else y
    disc_dist = y if y.discrete else x
    raw_support = disc_dist.get_support()
    
    if raw_support is not None:
        raw_pmf = disc_dist.func(raw_support)
        significant = raw_pmf > 1e-10
        cached_support = raw_support[significant]
        cached_pmf = raw_pmf[significant]

    def evaluate_mixed(z):
        if z < new_lower or z > new_upper: return 0.0
        shifted_z = z - cached_support
        valid_mask = (shifted_z >= cont_dist.lower_limit) & (shifted_z <= cont_dist.upper_limit)
        if not np.any(valid_mask): return 0.0
        return np.sum(cont_dist.func(shifted_z[valid_mask]) * cached_pmf[valid_mask])

    if not x.discrete and not y.discrete: eval_func = evaluate_continuous
    elif x.discrete and y.discrete: eval_func = evaluate_discrete
    else: eval_func = evaluate_mixed

    def vectorized_func(z_array):
        if np.isscalar(z_array): return eval_func(z_array)
        return np.array([eval_func(zi) for zi in np.atleast_1d(z_array)], dtype=float)

    return distribution(f"({x.name} + {y.name})", f"Sum of ({x.parameters}) and ({y.parameters})", 
                        vectorized_func, new_lower, new_upper, new_discrete, new_plot_lower, new_plot_upper)


def subtract_distributions(x, y):
    new_lower = x.lower_limit - y.upper_limit
    new_upper = x.upper_limit - y.lower_limit
    new_plot_lower = x.plot_lower - y.plot_upper
    new_plot_upper = x.plot_upper - y.plot_lower
    new_discrete = x.discrete and y.discrete

    def evaluate_continuous(z):
        if z < new_lower or z > new_upper: return 0.0
        lower_bound = max(x.lower_limit, z + y.lower_limit)
        upper_bound = min(x.upper_limit, z + y.upper_limit)
        if lower_bound > upper_bound: return 0.0
        res, _ = quad(lambda t: x.func(t) * y.func(t - z), lower_bound, upper_bound, limit=100)
        return res

    def evaluate_discrete(z):
        if z < new_lower or z > new_upper or not float(z).is_integer(): return 0.0
        prob_sum = 0.0
        for k in x.get_support():
            if y.lower_limit <= (k - z) <= y.upper_limit:
                prob_sum += x.func(k) * y.func(k - z)
        return prob_sum

    cont_dist = x if not x.discrete else y
    disc_dist = y if y.discrete else x
    raw_support = disc_dist.get_support()
    
    if raw_support is not None:
        raw_pmf = disc_dist.func(raw_support)
        significant = raw_pmf > 1e-10
        cached_support = raw_support[significant]
        cached_pmf = raw_pmf[significant]

    def evaluate_mixed(z):
        if z < new_lower or z > new_upper: return 0.0
        shifted_z = z + cached_support if y.discrete else cached_support - z
        valid_mask = (shifted_z >= cont_dist.lower_limit) & (shifted_z <= cont_dist.upper_limit)
        if not np.any(valid_mask): return 0.0
        return np.sum(cont_dist.func(shifted_z[valid_mask]) * cached_pmf[valid_mask])

    if not x.discrete and not y.discrete: eval_func = evaluate_continuous
    elif x.discrete and y.discrete: eval_func = evaluate_discrete
    else: eval_func = evaluate_mixed

    def vectorized_func(z_array):
        if np.isscalar(z_array): return eval_func(z_array)
        return np.array([eval_func(zi) for zi in np.atleast_1d(z_array)], dtype=float)

    return distribution(f"({x.name} - {y.name})", f"Diff of ({x.parameters}) and ({y.parameters})", 
                        vectorized_func, new_lower, new_upper, new_discrete, new_plot_lower, new_plot_upper)


def multiply_distributions(x, y):
    corners = []
    plot_corners = []
    for cx, px in zip([x.lower_limit, x.upper_limit], [x.plot_lower, x.plot_upper]):
        for cy, py in zip([y.lower_limit, y.upper_limit], [y.plot_lower, y.plot_upper]):
            if np.isinf(cx) or np.isinf(cy): corners.append(np.inf if np.sign(cx) == np.sign(cy) else -np.inf)
            else: corners.append(cx * cy)
            
            if np.isinf(px) or np.isinf(py): plot_corners.append(np.inf if np.sign(px) == np.sign(py) else -np.inf)
            else: plot_corners.append(px * py)
                
    new_lower, new_upper = min(corners), max(corners)
    new_plot_lower, new_plot_upper = min(plot_corners), max(plot_corners)
    new_discrete = x.discrete and y.discrete

    def evaluate_continuous(z):
        if z < new_lower or z > new_upper: return 0.0
        def integrand(t):
            if t == 0 or not (y.lower_limit <= (z / t) <= y.upper_limit): return 0.0
            return x.func(t) * y.func(z / t) * (1.0 / abs(t))
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            if x.lower_limit < 0 < x.upper_limit:
                res1, _ = quad(integrand, x.lower_limit, 0, limit=100)
                res2, _ = quad(integrand, 0, x.upper_limit, limit=100)
                return res1 + res2
            else:
                res, _ = quad(integrand, x.lower_limit, x.upper_limit, limit=100)
                return res

    def evaluate_discrete(z):
        if z < new_lower or z > new_upper or not float(z).is_integer(): return 0.0
        prob_sum = 0.0
        for k in x.get_support():
            if k == 0:
                if z == 0: prob_sum += x.func(0)
                continue
            if float(z / k).is_integer() and y.lower_limit <= (z / k) <= y.upper_limit:
                prob_sum += x.func(k) * y.func(z / k)
        return prob_sum

    cont_dist = x if not x.discrete else y
    disc_dist = y if y.discrete else x
    raw_support = disc_dist.get_support()
    
    if raw_support is not None:
        raw_pmf = disc_dist.func(raw_support)
        significant = (raw_pmf > 1e-10) & (raw_support != 0) 
        cached_support = raw_support[significant]
        cached_pmf = raw_pmf[significant]

    def evaluate_mixed(z):
        if z < new_lower or z > new_upper: return 0.0
        scaled_z = z / cached_support
        valid_mask = (scaled_z >= cont_dist.lower_limit) & (scaled_z <= cont_dist.upper_limit)
        if not np.any(valid_mask): return 0.0
        return np.sum(cont_dist.func(scaled_z[valid_mask]) * cached_pmf[valid_mask] / np.abs(cached_support[valid_mask]))

    if not x.discrete and not y.discrete: eval_func = evaluate_continuous
    elif x.discrete and y.discrete: eval_func = evaluate_discrete
    else: eval_func = evaluate_mixed

    def vectorized_func(z_array):
        if np.isscalar(z_array): return eval_func(z_array)
        return np.array([eval_func(zi) for zi in np.atleast_1d(z_array)], dtype=float)

    return distribution(f"({x.name} * {y.name})", f"Prod of ({x.parameters}) and ({y.parameters})", 
                        vectorized_func, new_lower, new_upper, new_discrete, new_plot_lower, new_plot_upper)