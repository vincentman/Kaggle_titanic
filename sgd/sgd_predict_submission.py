from common import process_data_from_Yassine
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

process_data = process_data_from_Yassine.ProcessData()
process_data.feature_engineering()
test_data = process_data.get_test_data()
x_test = test_data.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)

x_test = StandardScaler().fit_transform(x_test.values)

clf = joblib.load('sgd_dump.pkl')

pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": clf.predict(x_test).astype(int)}).to_csv(
    'submission.csv', header=True, index=False)
