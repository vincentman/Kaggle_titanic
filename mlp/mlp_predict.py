from common import process_data
import pandas as pd
from common import load_csv
from common import process_data_from_Yassine
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# x, y = load_csv.load_data(False)
# x_test = process_data.get_clean_data(x)
# x_test = x_test.drop(['Survived'], axis=1)

process_data = process_data_from_Yassine.ProcessData(train_data_ratio=0.7)
process_data.feature_engineering()
validation_data = process_data.get_validation_data()
y = validation_data.Survived
x_test = validation_data.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)
print('y.shape: ', y.shape)

x_test = StandardScaler().fit_transform(x_test.values)

model = load_model('mlp_train_model.h5')
scores = model.evaluate(x_test, y.values)
print('MLP, test score: ', scores[1])

with open('mlp_predict_info.txt', 'w') as file:
    file.write('test accuracy = {}\n'.format(scores[1]))
