# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from common import process_data
from common import statistics as stat
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.regularizers import l2
from common import load_csv
from common import process_data_from_Yassine
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# x, y = load_csv.load_data(True)
# x_train = process_data.get_clean_data(x)
# x_train = x_train.drop(['Survived'], axis=1)

process_data = process_data_from_Yassine.ProcessData(train_data_ratio=0.9)
process_data.feature_engineering()
train_data = process_data.get_train_data()
y = train_data.Survived
x_train = train_data.drop(['Survived'], axis=1)

# x_train = process_data.get_feature_importances(x_train.columns.values, x_train.values, y.values)
# stat.show_statistics(x)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

x_train = StandardScaler().fit_transform(x_train.values)

# x_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_train.values)

# regularizer = l2(0.02)
regularizer = None
neuron_units = 70
model = Sequential()
# model.add(Dense(units=40, input_dim=x_train.shape[1],
#                 kernel_initializer='uniform',
#                 activation='relu'))
# model.add(Dense(units=30,
#                 kernel_initializer='uniform',
#                 activation='relu'))
# model.add(Dense(units=1,
#                 kernel_initializer='uniform',
#                 activation='sigmoid'))
# 11+5 layers
model.add(Dense(units=50, input_dim=x_train.shape[1], kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(units=50, kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dense(units=neuron_units, kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Dropout(0.02))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

epochs = 30
adam = Adam(lr=0.01, decay=0.001, beta_1=0.9, beta_2=0.9)
model.compile(loss='binary_crossentropy',
              optimizer=adam, metrics=['accuracy'])
# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
start = time.time()
earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
callbacks = None
# callbacks = [earlyStopping]
train_history = model.fit(x=x_train,
                          y=y,
                          validation_split=0.1,
                          epochs=epochs,
                          shuffle=True,
                          batch_size=20, verbose=2,
                          callbacks=callbacks)
train_acc, validation_acc = stat.show_train_history(train_history, 'acc', 'val_acc', 'accuracy')
train_loss, validation_loss = stat.show_train_history(train_history, 'loss', 'val_loss', 'loss')

end = time.time()
elapsed_train_time = 'elapsed training time: {} min, {} sec '.format(int((end - start) / 60), int((end - start) % 60))
print(elapsed_train_time)

model.save('mlp_train_model.h5')

with open('mlp_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write('train accuracy = {}, validation accuracy = {}\n'.format(train_acc, validation_acc))
    file.write('train loss = {}, validation loss = {}\n'.format(train_loss, validation_loss))
