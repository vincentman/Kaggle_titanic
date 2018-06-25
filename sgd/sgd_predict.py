from common import process_data_from_Yassine
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

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
clf = joblib.load('sgd_dump.pkl')
test_score = clf.score(x_test, y_test)
print('SGD, test accuracy: ', test_score)
with open('sgd_predict_info.txt', 'w') as file:
    file.write('test accuracy = {}'.format(test_score))