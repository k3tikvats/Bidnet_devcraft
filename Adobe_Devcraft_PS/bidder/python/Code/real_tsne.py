import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic 2D data
np.random.seed(42)
high_variance_data = np.random.normal(loc=0, scale=5, size=(500, 2))  # High variance
targets_high = np.zeros(500)

low_variance_data = np.random.normal(loc=0, scale=2, size=(500, 2))  # Low variance
targets_low = np.ones(500)

# Combine data
X = np.vstack((high_variance_data, low_variance_data))
y = np.hstack((targets_high, targets_low))

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='after embeddings', alpha=0.7, c='blue')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Raw data', alpha=0.7, c='red')
plt.legend()
plt.title("Visualization ")
plt.show()
