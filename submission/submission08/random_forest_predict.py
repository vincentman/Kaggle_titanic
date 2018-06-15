from common import process_data
from common import process_train_test_data
import pandas as pd
from sklearn.externals import joblib
from common import load_csv

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# x, y = load_csv.load_data(False)
# x_test = process_data.get_clean_data(x)
# x_test = x_test.drop(['Survived'], axis=1)

all_data = process_train_test_data.get_clean_data()
validation_data = process_train_test_data.get_validation_data(all_data)
y = validation_data.Survived
x_test = validation_data.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)
print('y.shape: ', y.shape)

random_forest_clf = joblib.load('random_forest_dump.pkl')
test_score = random_forest_clf.score(x_test, y)
print('Random Forest, test accuracy: ', test_score)
with open('random_forest_predict_info.txt', 'w') as file:
    file.write('test accuracy = {}'.format(test_score))
