import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam,Adadelta,SGD,Adagrad,RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
import random
import math
import os
import re

path = "../basket.csv"
#/home/qwerty/qwerty/PaulusCereus/basket.csv
df = pd.read_csv(path, encoding= 'cp1251', sep=',', header=0, index_col=0)
print(df)
Xtrain = np.array(df[['COM 1','COM 2', 'ftime']].astype('int'))
Ytrain = np.array(df['fcount'].astype('int'))

x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.05)

model = Sequential()
model.add(Dense(200,input_dim=3, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="MAE", optimizer="adam")

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    verbose=1, shuffle=True)

data = np.array([[0, 2.0, 90.0]])
prediction = model.predict(data)

from keras2cpp import export_model
export_model(model, 'example.model')
