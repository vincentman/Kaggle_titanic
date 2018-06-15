import pandas as pd
import math

train_data_ratio = 0.9


def load_data(is_train):
    pd_csv = pd.read_csv('../train.csv')
    train_data_size = math.ceil(891 * train_data_ratio)
    if is_train:
        x = pd_csv.iloc[:train_data_size, :]
        y = pd_csv.Survived[:train_data_size]
    else:
        x = pd_csv.iloc[train_data_size:, :]
        y = pd_csv.Survived[train_data_size:]
    return x, y

