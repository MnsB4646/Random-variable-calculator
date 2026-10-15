import numpy as np
import scipy.stats as stats
from distribution import distribution

def Normal(mu=0, sigma=1):
    return distribution(
        name=f"N({mu}, {sigma})", 
        parameters=f"mu={mu}, sigma={sigma}", 
        func=stats.norm(loc=mu, scale=sigma).pdf, #type: ignore
        lower_limit=-np.inf, 
        upper_limit=np.inf, 
        discrete=False,
        plot_lower=mu - 4 * sigma,
        plot_upper=mu + 4 * sigma
    )

def Uniform(a=0, b=1):
    return distribution(
        name=f"U({a}, {b})", 
        parameters=f"a={a}, b={b}", 
        func=stats.uniform(loc=a, scale=b-a).pdf, #type: ignore
        lower_limit=a, 
        upper_limit=b, 
        discrete=False, 
        plot_lower=a,
        plot_upper=b
    )

def Poisson(mu=1):
    return distribution(
        name=f"Po({mu})", 
        parameters=f"m={mu}", 
        func=stats.poisson(mu=mu).pmf, #type: ignore
        lower_limit=0, 
        upper_limit=np.inf, 
        discrete=True, 
        plot_lower=0,
        plot_upper=int(stats.poisson(mu=mu).ppf(0.9999))
    )

def Binomial(n=1, p=0.5):
    return distribution(
        name=f"Bin({n}, {p})", 
        parameters=f"n={n}, p={p}", 
        func=stats.binom(n=n, p=p).pmf, #type: ignore
        lower_limit=0, 
        upper_limit=n, 
        discrete=True, 
        plot_lower=0,
        plot_upper=n
    )

def Cauchy(x0=0, gamma=1):
    return distribution(
        name=f"C({x0}, {gamma})", 
        parameters=f"x0={x0}, gamma={gamma}", 
        func=stats.cauchy(loc=x0, scale=gamma).pdf, #type: ignore
        lower_limit=-np.inf, 
        upper_limit=np.inf, 
        discrete=False, 
        plot_lower=x0 - 10 * gamma,
        plot_upper=x0 + 10 * gamma
    )

def Chi_square(df=1):
    return distribution(
        name=f"Chi_square({df})", 
        parameters=f"df={df}", 
        func=stats.chi2(df=df).pdf, #type: ignore
        lower_limit=0, 
        upper_limit=np.inf, 
        discrete=False, 
        plot_lower=0,
        plot_upper=stats.chi2(df=df).ppf(0.9999)
    )

def Exponential(lambda_=1):
    return distribution(
        name=f"Exp({lambda_})", 
        parameters=f"lambda={lambda_}", 
        func=stats.expon(scale=1/lambda_).pdf, #type: ignore
        lower_limit=0, 
        upper_limit=np.inf, 
        discrete=False, 
        plot_lower=0,
        plot_upper=stats.expon(scale=1/lambda_).ppf(0.9999)
    )

def Gamma(shape=1, scale=1):
    return distribution(
        name=f"Gamma({shape}, {scale})", 
        parameters=f"shape={shape}, scale={scale}", 
        func=stats.gamma(a=shape, scale=scale).pdf, #type: ignore
        lower_limit=0, 
        upper_limit=np.inf, 
        discrete=False, 
        plot_lower=0,
        plot_upper=stats.gamma(a=shape, scale=scale).ppf(0.9999)
    )

def Laplace(mu=0, b=1):
    return distribution(
        name=f"Laplace({mu}, {b})", 
        parameters=f"mu={mu}, b={b}", 
        func=stats.laplace(loc=mu, scale=b).pdf, #type: ignore
        lower_limit=-np.inf, 
        upper_limit=np.inf, 
        discrete=False, 
        plot_lower=mu - 10 * b,
        plot_upper=mu + 10 * b
    )