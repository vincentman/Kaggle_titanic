from common import process_data
from common import process_train_test_data
from common import process_data_from_Yassine
import pandas as pd
from sklearn.externals import joblib
import numpy as np

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# x = pd.read_csv('../test.csv').iloc[:, :]
# x_test = process_data.get_clean_data(x)

# all_data = process_train_test_data.get_clean_data()
# test_data = process_train_test_data.get_test_data(all_data)

process_data = process_data_from_Yassine.ProcessData()
process_data.feature_engineering()
test_data = process_data.get_test_data()
x_test = test_data.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)

random_forest_clf = joblib.load('random_forest_dump.pkl')

pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": random_forest_clf.predict(x_test).astype(int)}).to_csv(
    'submission.csv', header=True, index=False)
