import pandas as pd
from sklearn.preprocessing import StandardScaler
from common import process_data
from common import statistics as stat
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.regularizers import l2
from common import load_csv
from common import process_data_from_Yassine

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# x, y = load_csv.load_data(True)
# x_train = process_data.get_clean_data(x)
# x_train = x_train.drop(['Survived'], axis=1)

process_data = process_data_from_Yassine.ProcessData(train_data_ratio=0.7)
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
model.add(Dropout(0.4))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

epochs = 20
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
                          batch_size=20, verbose=2)
train_acc, validation_acc = stat.show_train_history(train_history, epochs, 'acc', 'val_acc', 'accuracy')
train_loss, validation_loss = stat.show_train_history(train_history, epochs, 'loss', 'val_loss', 'loss')

end = time.time()
elapsed_train_time = 'elapsed training time: {} min, {} sec '.format(int((end - start) / 60), int((end - start) % 60))
print(elapsed_train_time)

model.save('mlp_train_model.h5')

with open('mlp_train_info.txt', 'w') as file:
    file.write(elapsed_train_time+'\n')
    file.write('train accuracy = {}, validation accuracy = {}\n'.format(train_acc, validation_acc))
    file.write('train loss = {}, validation loss = {}\n'.format(train_loss, validation_loss))
