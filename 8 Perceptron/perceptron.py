# %%
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

# %%
iris = datasets.load_iris()
labels = iris.target
septal_lengths = iris.data[:, 0]
septal_width = iris.data[:, 1]

fig, ax = plt.subplots()
for idx, color in enumerate(['red', 'green', 'blue']):
    ax.scatter(septal_lengths[labels == idx], septal_width[labels == idx], c=color, label=iris.target_names[idx],
               alpha=0.3, edgecolors='none')
ax.legend()
plt.show()
# %%
