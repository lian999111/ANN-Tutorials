from print_callback import PrintCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
from keras.optimizers import Adam
import numpy as np

import io

#open the file
path = 'faust.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

#Look at the text characteristics to get an idea of the text
print('-----------Text characteristics--------')
print('Text length:', len(text))
chars = sorted(list(set(text)))
print('Number of unique characters in text:', len(chars))

#SOLUTION 2a
#map characters to indices and vice-versa
char2int = dict((c, i) for i, c in enumerate(chars))
int2char = dict((i, c) for i, c in enumerate(chars))

len_seq = 50
step = 3
sequences = []
targets = []
for i in range(0, len(text) - len_seq, step):
    sequences.append(text[i: i + len_seq])
    targets.append(text[i + len_seq])
print('Number of training sequences:', len(sequences))

#SOLUTION 2b
x = np.zeros((len(sequences), len_seq, len(chars)))
y = np.zeros((len(sequences), len(chars)))

#compute one-hot encoding
for i, sentence in enumerate(sequences):
    for t, char in enumerate(sentence):
        x[i, t, char2int[char]] = 1
    y[i, char2int[targets[i]]] = 1
    
#SOLUTION 2c
#Build the model
model = Sequential()
model.add(SimpleRNN(256, input_shape=(len_seq, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#SOLUTION 2d
#Train the model
print_callback = PrintCallback(text, chars, int2char, char2int, model)
model.fit(x, y,
          batch_size=128,
          epochs=20,
          callbacks=[print_callback])
model.save_weights('weights_lstm.h5f')
