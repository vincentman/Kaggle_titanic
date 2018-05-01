import pandas as pd
import math

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

pd_csv = pd.read_csv('train.csv')
data_size = math.ceil(891*0.7)
x = pd_csv.iloc[:data_size, :]
y = pd_csv.Survived[:data_size]
# print('x.shape: ', x.shape)
# print('x.columns => \n', x.columns.values)

from process_data import get_clean_data
x_train = get_clean_data(x)
# print(x_train.describe())

# from statistics import show_statistics
# show_statistics(x)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train.values)
