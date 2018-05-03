from common import process_data
import pandas as pd
from sklearn.externals import joblib
import numpy as np

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x = pd.read_csv('../test.csv').iloc[:, :]

x_test = process_data.get_clean_data(x)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)

svm_clf = joblib.load('svm_dump.pkl')

pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": svm_clf.predict(x_test)}).to_csv(
    'submission.csv', header=True, index=False)
