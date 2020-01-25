from Perceptron import *


# Load the dataset
iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, (0, 3)]
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add x0 = 1 for bias term
Y = np.where(iris_dataset.target == 2, 1, -1)

# Display the dataset
fig, ax = plt.subplots()
ax.scatter(X[Y == 1, 1], X[Y == 1, 2], label="virginica")
ax.scatter(X[Y == -1, 1], X[Y == -1, 2], label="other")
ax.set_xlabel("Septal length")
ax.set_ylabel("Petal width")
ax.legend()

# Train the pocket perceptron
model = PocketPerceptron()
model.train(X, Y, learning_rate=1, max_epochs=700, max_error=0, shuffle=False)
print(f"{model.smallest_training_errors[-1]} errors in the training set")
model.plot_2D_decision_boundary(ax, X)
fig.savefig("figures/virginica-scatter.png", dpi=200)
model.plot_training_errors()
plt.savefig("figures/virginica-error.png", dpi=200)
plt.show()
