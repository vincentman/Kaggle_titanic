from sklearn.svm import SVC
from common import process_data
from common import process_train_test_data
from common import process_data_from_Yassine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv

# x, y = load_csv.load_data(False)
# x_test = process_data.get_clean_data(x)
# x_test = x_test.drop(['Survived'], axis=1)

# all_data = process_train_test_data.get_clean_data()
# validation_data = process_train_test_data.get_validation_data(all_data)

process_data = process_data_from_Yassine.ProcessData()
process_data.feature_engineering()
validation_data = process_data.get_validation_data()
y = validation_data.Survived
x_test = validation_data.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)
print('y.shape: ', y.shape)

x_test = StandardScaler().fit_transform(x_test.values)
y_test = y.values
svm = joblib.load('svm_dump.pkl')
test_score = svm.score(x_test, y_test)
print('SVM, test accuracy: ', test_score)
with open('svm_predict_info.txt', 'w') as file:
    file.write('test accuracy = {}'.format(test_score))