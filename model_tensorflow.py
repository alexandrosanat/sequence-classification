import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import backend as K


data = pd.read_csv('./data/preprocessed/ACPs_Breast_cancer.csv')
data['sequence'] = data['sequence'].map(lambda x: [str(y) for y in x])  # Convert to lists of characters

word_dictionary = {y: x for x, y in enumerate(list(set(data['sequence'].sum())))}
data['sequence'] = data['sequence'].map(lambda x: [word_dictionary[y] for y in x])
max_sequence_length = max(data['sequence'].apply(len))


y = data.pop('class')
X = data

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.4, random_state=1, stratify=y)

X_train = sequence.pad_sequences(np.squeeze(X_train), maxlen=max_sequence_length)
X_test = sequence.pad_sequences(np.squeeze(X_test), maxlen=max_sequence_length)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


top_words = len(word_dictionary)
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=64)

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 38, 32)            704       
_________________________________________________________________
lstm (LSTM)                  (None, 100)               53200     
_________________________________________________________________
dense (Dense)                (None, 1)                 101       
=================================================================
Total params: 54,005
Trainable params: 54,005
Non-trainable params: 0
_________________________________________________________________
"""

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("AUC: %.2f%%" % (scores[1]*100))

predictions = np.where(model.predict(X_test) > 0.5, 1, 0)
targets = y_test.values.reshape(-1, 1)

np.concatenate((predictions, targets), axis=1)

print(classification_report(targets, predictions))
