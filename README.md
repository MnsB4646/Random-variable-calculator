# Random-variable-calculator

A program for combining and visualizing probability distributions.

This calculator supports two distinct execution paths: an **Analytical Mode** that uses exact numerical calculus (via SciPy integration) to compute precise continuous and discrete convolutions, and a **Monte Carlo Mode** that uses grid approximations and random sampling to compute conditional probabilities (e.g., $X \vert{} X > Y$) efficiently.

## Features

* **Operator Overloading:** Combine distributions using standard Python math operators (`Z = X + Y`, `Z = X * Y`). The underlying classes automatically handle the continuous convolution integrals or discrete PMF summations.
* **Two Execution Modes:** Switch between analytical calculations and Monte Carlo array simulations by toggling the variable USE_MONTE_CARLO in main.py.
* **Robust Numerical Safety:** Built-in dynamic plot boundaries and grid generation bounds prevent integration failures and `NaN` errors when dealing with infinite tails or floating-point inaccuracies.
* **Conditional Probability:** Easily model complex dependent scenarios by passing Monte Carlo arrays through boolean masks.

## Requirements

* Python 3.x
* `numpy`
* `scipy`
* `matplotlib`

## Project Structure

* `distribution.py`: Contains the `distribution` class and the convolution logic for adding, subtracting, and multiplying probability functions.
* `models.py`: A library of functions (e.g., `Normal`, `Poisson`, `Binomial`, `Uniform`, `Laplace`) that wrap SciPy stats functions into `distribution` objects with safe limits.
* `main.py`: The execution environment containing the visualization functions (`plot_distribution`, `plot_monte_carlo`) and the Monte Carlo `generate_samples` helper.

## Usage

Configure your variables and operations directly inside the `main()` function in `main.py`. You can select the computation method using the `USE_MONTE_CARLO` boolean toggle:

* Set `USE_MONTE_CARLO = False` to compute exact analytical curves.
* Set `USE_MONTE_CARLO = True` to run sample-based simulations.

### 1. Exact Analytical Mode

When you need mathematically precise probability density functions or cumulative distribution functions, use the analytical mode.

```python

#Define base distributions
X = Poisson(mu=5)
Y = Binomial(n=10, p=0.5)

#Perform exact convolution
Z = X + Y

#Visualize the exact PDF/PMF and CDF
plot_distribution(
    main_dist=Z, 
    components=[X, Y], 
    title="Analytical: Z = X + Y",
    xlim=(-50, 20) 
)

```

### 2. Monte Carlo Simulation Mode

When dealing with conditional logic this is much faster. Monte Carlo mode bypasses integration by generating large, statistically representative sample arrays using grid approximation.

```python

#Define variables using existing distribution models
X = Normal(mu=0, sigma=1)
Y = Normal(mu=0, sigma=2)

#Generate arrays of 1 million samples
N = 1_000_000
x_samples = generate_samples(X, size=N)
y_samples = generate_samples(Y, size=N)

#Perform chained operations instantly using vector math
z_samples = x_samples + y_samples

#Apply conditional masking (e.g., Z given that Z > Y)
valid_events_mask = z_samples > y_samples
z_given_z_gt_y = z_samples[valid_events_mask]

#Plot the overlapping arrays
plot_monte_carlo(
    main_samples=z_given_z_gt_y, 
    main_label='Z Conditioned on Z > Y',
    components=[
        (x_samples, 'X (Normal)'),
        (y_samples, 'Y (Normal)'),
        (z_samples, 'Z (X + Y)')
    ],
    title="Monte Carlo Simulation"
)

```

## Adding Custom Distributions

You can easily expand the calculator by adding new functions to `models.py`.

* **`lower_limit` / `upper_limit**`: The strict mathematical support. Use `-np.inf` or `np.inf` if the tails are unbounded. This tells the exact calculus solver where to integrate.
* **`plot_lower` / `plot_upper**`: Finite boundaries used for graphing and generating the Monte Carlo sampling grid. For infinite distributions, tie this to the statistical parameters (e.g., $\mu \pm 4\sigma$ for a Normal distribution) to capture 99.99% of the relevant area without causing grid cutoff errors.