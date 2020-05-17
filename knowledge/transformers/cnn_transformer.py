from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Conv1D, Embedding, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

class CNNTransformer:
    def __init__(self):
        self.model = self.model()
        self.model_truncated = None

    def transform(self, sentence):
        return sentence

    def fit(self, data_x, data_y):
        print(data_x.shape)
        self.model.fit(data_x, data_y, epochs=200)
        self.model_truncated = self.truncated()

    def truncated(self):
        model_truncated = Sequential()
        model.add(Conv1D(25, 5, padding='valid', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Conv1D(50, 3, padding='valid', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model_truncated.add(Dense(40, activation='relu'))

        for i, layer in enumerate(model_truncated.layers):
            layer.set_weights(self.model.layers[i].get_weights())

        model_truncated.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model_truncated
    
    def model(self):
        model = Sequential()
        model.add(Conv1D(25, 5, padding='valid', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Conv1D(50, 3, padding='valid', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(40, activation='relu'))
        model.add(Dense(18, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def f1_score(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    
    def dict_to_dataframe(self, dct):
        data = pd.DataFrame(columns=['paths'])
        for idx in range(0, len(dct)):
            data.loc[idx] = [dct[idx]['embed']]
        data = pad_sequences(data['paths'], padding='post', maxlen=10)
        print('Converted!')
        return data
