# %%
import tensorflow as tf 
import numpy as n1p 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
def prepareData(time_steps, num_samples):
    x = np.linspace(0, 100, num_samples + time_steps)
    f = np.sin(x) + 0.1 * x

    x_samples = np.zeros((num_samples, time_steps))
    y_samples = np.zeros(num_samples)
    for idx, y_sample in enumerate(f[time_steps:]):
        x_samples[idx, :] = f[idx : idx+time_steps]
        y_samples[idx] = y_sample

    return x_samples, y_samples

# %%
if __name__ == '__main__':
    # Prepare data
    time_steps = 1
    num_samples = 1000
    x, y = prepareData(time_steps=time_steps, num_samples=num_samples)

    train_idc = np.random.randint(0, num_samples, num_samples*0.7)
    x_train = x[0:int(num_samples*0.7)]
    y_train = y[0:int(num_samples*0.7)]
    x_test = x[int(num_samples*0.7):]
    y_test = y[int(num_samples*0.7):]

    simple_rnn = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=(time_steps,)),
        tf.keras.layers.SimpleRNN(32, return_sequences=False),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    simple_rnn.summary()
    simple_rnn.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = simple_rnn.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# %%
y_pred_train = simple_rnn.predict(x_train)
y_pred_test = simple_rnn.predict(t_tests)
# %%
