import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import process_data
import statistics as stat
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.regularizers import l1, l2, l1_l2

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

pd_csv = pd.read_csv('train.csv')
data_size = math.ceil(891 * 0.7)
x = pd_csv.iloc[:data_size, :]
y = pd_csv.Survived[:data_size]
# print('x.shape: ', x.shape)
# print('x.columns => \n', x.columns.values)

x_train = process_data.get_clean_data(x)
# print(x_train.describe())

# x_train = process_data.get_feature_importances(x_train.columns.values, x_train.values, y.values)
# x_train = x_train.get(['Pclass', 'Sex', 'Age', 'SibSp'])
# x_train = x_train.get(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
# stat.show_statistics(x)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

x_train = StandardScaler().fit_transform(x_train.values)
y_train = y.values

model = Sequential()
model.add(Dense(units=30, input_dim=x_train.shape[1], kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dense(units=30, input_dim=9,
#                 # kernel_initializer='uniform',
#                 kernel_regularizer=l2(0.01),
#                 activation='relu'))
model.add(Dense(units=30, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(units=30, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(units=30, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

epochs = 30
# from keras.optimizers import Adam
# learning_rate = 0.001
# adam = Adam(lr=learning_rate, decay=0.0001)
# model.compile(loss='binary_crossentropy',
#               optimizer=adam, metrics=['accuracy'])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
start = time.time()
train_history = model.fit(x=x_train,
                          y=y,
                          validation_split=0.1,
                          epochs=epochs,
                          shuffle=True,
                          batch_size=24, verbose=2)
stat.show_train_history(train_history, epochs, 'acc', 'val_acc', 'accuracy')
stat.show_train_history(train_history, epochs, 'loss', 'val_loss', 'loss')

end = time.time()
elapsed_train_time = 'elapsed training time: {} min, {} sec '.format(int((end - start) / 60), int((end - start) % 60))
print(elapsed_train_time)
