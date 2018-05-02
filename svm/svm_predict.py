from sklearn.svm import SVC
from common import process_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv

x, y = load_csv.load_data(False)

x_test = process_data.get_clean_data(x)

print('x_train.shape: ', x_test.shape)
print('x_train.columns => \n', x_test.columns.values)
print('y.shape: ', y.shape)

x_test = StandardScaler().fit_transform(x_test.values)
y_test = y.values
svm = joblib.load('svm_dump.pkl')
print('SVM, test accuracy: ', svm.score(x_test, y_test))