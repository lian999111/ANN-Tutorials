import os
from Perceptron import *


# Create a folder for saving the figures
if not os.path.exists("figures"):
    os.makedirs("figures")

# Load the dataset
iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, :2]  # we only take the first two features
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add x0 = 1 for bias term
Y = np.where(iris_dataset.target == 0, 1, -1) # reassign labels

# Display the dataset
fig, ax = plt.subplots()
ax.scatter(X[Y == 1, 1], X[Y == 1, 2], label="setosa")
ax.scatter(X[Y == -1, 1], X[Y == -1, 2], label="other")
ax.set_xlabel("Septal length")
ax.set_ylabel("Septal width")
ax.legend()

# Train the basic perceptron
model = Perceptron()
model.train(X, Y, learning_rate=1, max_epochs=1000, max_error=0, shuffle=False)
print(f"{model.count_errors(X, Y)} classification errors in training set")
model.plot_2D_decision_boundary(ax, X)
fig.savefig("figures/setosa-scatter.png", dpi=200)
model.plot_training_errors()
plt.savefig("figures/setosa-error.png", dpi=200)
plt.show()