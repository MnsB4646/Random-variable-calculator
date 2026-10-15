import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from models import Normal, Uniform, Poisson, Binomial, Cauchy, Chi_square, Exponential, Gamma, Laplace

def plot_distribution(main_dist, components=None, title="Distribution", xlim=None):
    if xlim is not None:
        plot_min, plot_max = xlim
        padding = 0
    else:
        plot_min = main_dist.plot_lower if main_dist.plot_lower != -np.inf else -10
        plot_max = main_dist.plot_upper if main_dist.plot_upper != np.inf else 10
        padding = (plot_max - plot_min) * 0.1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    dists_to_plot = [main_dist]
    if components:
        dists_to_plot.extend(components)
        
    colors = ['purple', 'teal', 'darkorange', 'crimson', 'forestgreen', 'gold', 'navy', 'maroon', 'olive', 'cyan']
    
    for i, dist in enumerate(dists_to_plot):
        color = colors[i % len(colors)]
        linewidth = 2.5 if i == 0 else 1.5
        alpha_fill = 0.2 if i == 0 else 0.0
        
        if dist.discrete:
            x = np.arange(int(plot_min), int(plot_max) + 1)
            y = dist.func(x)
            cdf_y = np.cumsum(y)
            
            markerline, stemlines, _ = ax1.stem(x, y, basefmt=" ", label=f"{dist.name} PMF")
            plt.setp(markerline, color=color, markersize=6 if i == 0 else 4)
            plt.setp(stemlines, color=color, linewidth=linewidth, alpha=0.7)
            
            ax2.step(x, cdf_y, where='post', color=color, linewidth=linewidth, label=f"{dist.name} CDF")
        else:
            x = np.linspace(plot_min - padding, plot_max + padding, 1000)
            y = dist.func(x)
            cdf_y = cumulative_trapezoid(y, x, initial=0)
            
            ax1.plot(x, y, color=color, linewidth=linewidth, label=f"{dist.name} PDF")
            if alpha_fill > 0:
                ax1.fill_between(x, y, alpha=alpha_fill, color=color)
            
            ax2.plot(x, cdf_y, color=color, linewidth=linewidth, label=f"{dist.name} CDF")

    ax1.set_ylabel("Probability Density / Mass")
    ax2.set_ylabel("Cumulative Probability")
    
    for ax in (ax1, ax2):
        ax.set_xlabel("Value (x)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend()
        
    plt.tight_layout()
    plt.show()

def plot_monte_carlo(main_samples, main_label, components=None, title="Monte Carlo Simulation", bins=150):
    plt.figure(figsize=(10, 6))
    colors = ['purple', 'teal', 'darkorange', 'crimson', 'forestgreen', 'gold', 'navy', 'maroon', 'olive', 'cyan']

    dists_to_plot = [(main_samples, main_label)]
    if components:
        dists_to_plot.extend(components)
        
    for i, (data, label) in enumerate(dists_to_plot):
        color = colors[i % len(colors)]
        
        if i == 0:
            # The primary result gets a filled histogram
            plt.hist(data, bins=bins, density=True, color=color, alpha=0.6, label=label)
        else:
            # All background components get hollow outlines
            plt.hist(data, bins=bins, density=True, color=color, histtype='step', linewidth=2.5, label=label)
            
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def generate_samples(dist, size=1_000_000, resolution=10000):
    """Generates Monte Carlo arrays directly from a distribution's func attribute."""
    #Use absolute limits if finite, otherwise fall back to visual plot limits
    low = dist.lower_limit if dist.lower_limit != -np.inf else dist.plot_lower
    high = dist.upper_limit if dist.upper_limit != np.inf else dist.plot_upper
    
    if dist.discrete:
        x_grid = np.arange(int(low), int(high) + 1)
    else:
        x_grid = np.linspace(low, high, resolution)
        
    #Evaluate the density at every point
    weights = dist.func(x_grid)
    
    # Normalize the densities into valid array probabilities
    weights = np.maximum(weights, 0)
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        raise ValueError(f"Grid evaluation for {dist.name} resulted in 0 probability. Check limits.")
        
    probabilities = weights / total_weight
    
    #Draw 'size' random samples from the grid based on the probabilities
    return np.random.choice(x_grid, size=size, p=probabilities)

def main():
    USE_MONTE_CARLO = True 
    
    if USE_MONTE_CARLO:
        print("Running in Monte Carlo mode")
        
        #Define variables using existing distributions
        X = Normal(mu=0, sigma=1)
        Y = Normal(mu=0, sigma=2)
        
        #Generate samples for each distribution
        N = 1_000_000
        x_samples = generate_samples(X, size=N)
        y_samples = generate_samples(Y, size=N)
        
        #Perform operations
        z_samples = x_samples + y_samples
        
        #Apply conditional masking
        valid_events_mask = z_samples > y_samples
        z_given_z_gt_y = z_samples[valid_events_mask]
        
        # Plot the results
        plot_monte_carlo(
            main_samples=z_given_z_gt_y, 
            main_label='Z Conditioned on Z > Y',
            components=[
                (x_samples, 'X (Normal)'),
                (y_samples, 'Y (Uniform)'),
                (z_samples, 'Z (X + Y)')
            ],
            title="Monte Carlo Simulation"
        )
    else:
        print(f"Running in analytical mode")
        #Define variables using existing distributions
        X = Poisson(mu=5)
        Y = Binomial(n=10, p=0.5)

        #Perform operations
        Z = X + Y
        
        #Plot the results
        plot_distribution(
            main_dist=Z, 
            components=[X, Y], 
            title="Analytical: Z = X + Y",
            xlim=(-50, 20) #Set to None for automatic limits
        )

if __name__ == "__main__":
    main()