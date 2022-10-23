from asyncio import base_tasks
from audioop import rms
import enum
import random
from tabnanny import verbose
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c,i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

'''
sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
# Target data
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sent in enumerate(sentences):
    for j, char in enumerate(sent):
        x[i, j, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1
'''

# build and compile model
'''
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(layers.Dense(len(characters)))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)
model.save('textgenerator.model')
'''

model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for j, char in enumerate(sentence):
            x[0, j, char_to_index[char]] = 1
        prediction = model.predict(x, verbose=0)[0]
        next_index = sample(prediction, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return (generated)

print('---------Temperature: 0.2---------')
print(generate_text(300, 0.2))
print('---------Temperature: 0.4---------')
print(generate_text(300, 0.4))
print('---------Temperature: 0.6---------')
print(generate_text(300, 0.6))
print('---------Temperature: 0.8---------')
print(generate_text(300, 0.8))