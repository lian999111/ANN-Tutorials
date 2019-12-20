# %%
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# %% Load fashion mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reduce training dataset
num_sample = 20000
sample_idc = np.random.choice(x_train.shape[0], num_sample, replace=False)
x_train = x_train[sample_idc]
y_train = y_train[sample_idc]

# Show a sample image
sample_idx = 0
sample_img = x_train[sample_idx, :]
plt.imshow(sample_img, cmap='gray')
plt.show()

# Normalize data
x_train = x_train/255
x_test = x_test/255

# %% Create a FCN
input_shape = x_train.shape[1:]
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=['accuracy'])

# %% Train
model.fit(x=x_train, y=y_train, batch_size=None, epochs=5, validation_data=(x_test, y_test))

# %%
test_idx = 1
test_img = x_test[test_idx, :]
# y = model(np.reshape(test_img, (1,)+test_img.shape))
# y_hat = tf.math.argmax(model(np.reshape(test_img, (1,)+test_img.shape)), axis=1)
y_hat = model.predict(np.reshape(test_img, (1,)+test_img.shape))
# print(y_hat.numpy())
print(np.argmax(y_hat))
print(y_test[test_idx])

# %%
x_test_ankel_boots = x_test[y_test == 9]
y_test_ankel_boots = y_test[y_test == 9]

for x, y in zip(x_test_ankel_boots, y_test_ankel_boots):
    y_hat = np.argmax(model.predict(np.reshape(x, (1,)+x.shape)))
    if y_hat != y:
        plt.imshow(x, cmap='gray')
        print('Wrongly classified as {}'.format(y_hat))
        plt.show()


# %%
