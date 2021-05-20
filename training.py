import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

df =  pd.read_csv('/content/drive/MyDrive/XRP/train_dataset1.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)

close_data = df['Closing Price (USD)'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.60
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

look_back = 20

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import losses, optimizers

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(optimizer=optimizers.Adam() , loss=losses.mean_squared_error)

num_epochs = 80
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

