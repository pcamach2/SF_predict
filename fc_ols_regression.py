import statsmodels.api as sm
import numpy as np

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
design_matrix = sm.add_constant(predictors)  # Add constant term

# Perform hierarchical regression
model = sm.OLS(clinical_outcome, design_matrix)
results = model.fit()

# Print regression summary
print(results.summary())