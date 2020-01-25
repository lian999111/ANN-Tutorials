from Perceptron import *


# Study the effect of shuffling on the basic perceptron

iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, :2]  # we only take the first two features
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add x0 = 1 for bias term
Y = np.where(iris_dataset.target == 0, 1, -1)  # reassign labels

model = Perceptron()
model.train(X, Y, learning_rate=1, max_epochs=1000, max_error=0, shuffle=False)
conv_unshuffled = (len(model.training_errors))
conv_shuffled = []
conv_shuffled_once = []
N = 50

for i in range(N):
    print(f"Run number {i}")
    model.train(X, Y, learning_rate=1, max_epochs=1000, max_error=0, shuffle=True)
    conv_shuffled.append(len(model.training_errors))
    shuffled_indices = np.random.permutation(len(X))
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]
    model.train(X, Y, learning_rate=1, max_epochs=1000, max_error=0, shuffle=False)
    conv_shuffled_once.append(len(model.training_errors))

fig, ax = plt.subplots(constrained_layout=True)
ax.bar(1, conv_unshuffled, color='b')
ax.bar(2, np.mean(conv_shuffled_once), yerr=np.std(conv_shuffled_once), color='orange')
ax.bar(3, np.mean(conv_shuffled), yerr=np.std(conv_shuffled), color='g')
ax.set_xticks(np.arange(1, 4))
ax.set_xticklabels(['Unshuffled', 'Shuffled once', 'Shuffled in each epoch'])
ax.set_ylabel('Iterations before convergence')
fig.suptitle(f"Classification error (mean and std over {N} runs)")
fig.savefig("figures/shuffled-basic.png", dpi=200)


# Study the effect of shuffling on pocket perceptron

iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, (0, 3)]
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add x0 = 1 for bias term
Y = np.where(iris_dataset.target == 2, 1, -1)

N = 50
num_epochs = 20

model_unshuffled = PocketPerceptron()
model_unshuffled.train(X, Y, learning_rate=1, max_epochs=num_epochs,
                       max_error=0, shuffle=False)
unshuffled = model_unshuffled.smallest_training_errors

shuffled = []
shuffled_once = []

for i in range(N):
    print(f"Run number {i}")
    model_shuffled = PocketPerceptron()
    model_shuffled_once = PocketPerceptron()
    model_shuffled.train(X, Y, learning_rate=1, max_epochs=num_epochs, max_error=0)
    shuffled.append(model_shuffled.smallest_training_errors)
    shuffled_indices = np.random.permutation(len(X))
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]
    model_shuffled_once.train(X, Y, learning_rate=1, max_epochs=num_epochs,
                              max_error=0, shuffle=False)
    shuffled_once.append(model_shuffled_once.smallest_training_errors)

shuffled_mean = np.mean(shuffled, axis=0)
shuffled_std = np.std(shuffled, axis=0)
shuffled_once_mean = np.mean(shuffled_once, axis=0)
shuffled_once_std = np.std(shuffled_once, axis=0)
fig, ax = plt.subplots()
ax.plot(unshuffled, color='b', label="Unshuffled")
ax.fill_between(np.arange(shuffled_once_mean.size),
                shuffled_once_mean - shuffled_once_std,
                shuffled_once_mean + shuffled_once_std,
                facecolor='orange', alpha=0.4)
ax.plot(shuffled_once_mean, color='orange', label="Shuffled once")
ax.fill_between(np.arange(shuffled_mean.size), shuffled_mean - shuffled_std,
                shuffled_mean + shuffled_std,
                facecolor='g', alpha=0.4)
ax.plot(shuffled_mean, color='g', label="Shuffled in each epoch")
ax.set_xlabel("Iterations")
ax.set_ylabel("Error")
ax.legend()
fig.suptitle(f"Classification error (mean and std over {N} runs)")
fig.savefig("figures/shuffled-pocket.png", dpi=200)

plt.show()
