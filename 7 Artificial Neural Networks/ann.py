# %%
import numpy as np
import matplotlib.pyplot as plt
import utils


x = np.linspace(0, 1, num=1000)
# %% Fig 1a
edges = [0.4, 0.6]
heights = [0.8]
W, b, v = utils.gen_params(edges, heights)

y = utils.f(W, b, v, x)
plt.plot(x, y)
plt.show()

# %% Fig 1b
edges = [0.2, 0.4, 0.6, 0.8]
heights = [0.5, 0, 0.2]
W, b, v = utils.gen_params(edges, heights)

y = utils.f(W, b, v, x)
plt.plot(x, y)
plt.show()

# %% Fig 1c
edges = [0.2, 0.4, 0.6, 0.8]
heights = [0.5, 0.8, 0.2]
W, b, v = utils.gen_params(edges, heights)

y = utils.f(W, b, v, x)
plt.plot(x, y)
plt.show()

# %% Approximate g(x) = sin(2*pi*x)
g = utils.g(x)

N = 20
edges = np.linspace(0, 1, num=N+1)
heights = utils.g(edges[:-1])
W, b, v = utils.gen_params(edges, heights)
y = utils.f(W, b, v, x)

plt.plot(x, g)
plt.plot(x, y)
plt.show()


# %%
