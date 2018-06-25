import pandas as pd
from common import process_data_from_Yassine
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

process_data = process_data_from_Yassine.ProcessData()
process_data.feature_engineering()
test_data = process_data.get_test_data()
x_test = test_data.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)

x_test = StandardScaler().fit_transform(x_test.values)

model = load_model('mlp_train_model.h5')

pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": model.predict_classes(x_test).ravel()}).to_csv(
    'submission.csv', header=True, index=False)
