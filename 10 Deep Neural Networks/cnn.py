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

x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))

# %% Create a CNN
input_shape = x_train.shape[1:]
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu'),
    
    # tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=['accuracy'])

# %% Train
model.fit(x=x_train, y=y_train, batch_size=None, epochs=5, validation_data=(x_test, y_test))

# %%
test_idx = 17
test_img = x_test[test_idx, :]
# y = model(np.reshape(test_img, (1,)+test_img.shape))
# y_hat = tf.math.argmax(model(np.reshape(test_img, (1,)+test_img.shape)), axis=1)
y_hat = model.predict(np.reshape(test_img, (1,)+test_img.shape))
# print(y_hat.numpy())
print(np.argmax(y_hat))
print(y_test[test_idx])

# %%
