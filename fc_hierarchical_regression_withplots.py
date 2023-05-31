import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generate sample data
n_samples = 100
n_edges = 500

# Simulate functional connectomes
connectome_1 = np.random.rand(n_samples, n_edges)
connectome_2 = np.random.rand(n_samples, n_edges)

# Simulate clinical outcome variable
clinical_outcome = np.random.rand(n_samples)

# Create a design matrix with predictors
predictors = np.concatenate((connectome_1, connectome_2), axis=1)

# Define the hierarchical model
with pm.Model() as hierarchical_model:
    # Priors for the model parameters
    alpha = pm.Normal("alpha", mu=0, sd=1)
    beta = pm.Normal("beta", mu=0, sd=1, shape=predictors.shape[1])

    # Linear regression equation
    mu = alpha + pm.math.dot(predictors, beta)

    # Likelihood function
    sigma = pm.HalfNormal("sigma", sd=1)
    clinical_outcome_obs = pm.Normal("clinical_outcome_obs", mu=mu, sd=sigma, observed=clinical_outcome)

    # Run the MCMC sampling
    trace = pm.sample(2000, tune=1000)

# Print the summary of the posterior distribution
pm.summary(trace)

# Plotting posterior distributions
pm.plot_posterior(trace["alpha"], credible_interval=0.95)
plt.xlabel("Alpha")
plt.ylabel("Density")
plt.title("Posterior Distribution of Alpha")
plt.show()

for i in range(predictors.shape[1]):
    pm.plot_posterior(trace["beta"][:, i], credible_interval=0.95)
    plt.xlabel("Beta")
    plt.ylabel("Density")
    plt.title(f"Posterior Distribution of Beta {i+1}")
    plt.show()

# Plotting trace plots
pm.traceplot(trace)
plt.show()