import numpy as np
import matplotlib.pyplot as plt

# Generate samples
sample_x = 2*np.random.rand(1000) - 1
sample_y = 2*np.random.rand(1000) - 1

# Compute Euclidean distances to the center of the square area
euclidean_dists = (sample_x**2 + sample_y**2)**0.5
plt.hist(euclidean_dists)
plt.show()

hist_of_dist, _ = np.histogram(euclidean_dists)
plt.subplot(1, 3, 1)
plt.plot(hist_of_dist)

# Generate normal distribution
normal_samples = np.random.randn(1000)
hist_of_normal, _ = np.histogram(normal_samples)
plt.subplot(1, 3, 2)
plt.plot(hist_of_normal)
plt.show()

# Compute quantiles
np.quantile()