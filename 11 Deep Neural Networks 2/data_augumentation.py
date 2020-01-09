# %%
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import matplotlib.pyplot as plt

# %% Load fashion mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reduce training dataset
num_sample = 20000
x_train = x_train[0:num_sample]
y_train = y_train[0:num_sample]

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

# %% Data augumentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.5,
    zoom_range=(0.9, 1.1),
    fill_mode='constant')

datagen.fit(x_train)

num_iters = 20000/64
x_aug = np.empty((0,)+x_train.shape[1:])
y_aug = np.empty((0,))
for count, (x_batch, y_batch) in enumerate(datagen.flow(x_train, y_train, batch_size=64)):
    x_aug = np.append(x_aug, x_batch, axis=0)
    y_aug = np.append(y_aug, y_batch, axis=0)
    if count > num_iters:
        break

# %% Divide data
x_train_pret = x_aug[:len(x_aug)//2]
y_train_pret = y_aug[:len(x_aug)//2]

x_test_pret = x_aug[len(x_aug)//2:]
y_test_pret = y_aug[len(x_aug)//2:]

# %% Create a CNN
input_shape = x_train.shape[1:]
model = tf.keras.models.Sequential([    
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10)
])

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=['accuracy'])

# %% Train
epochs = 1
history = model.fit(x=x_train, y=y_train, batch_size=None, epochs=epochs, validation_data=(x_test, y_test))
model.save('model_for_problem_3')

# %% Evaluate

