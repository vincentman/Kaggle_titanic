from common import process_data
import pandas as pd
from common import load_csv
from keras.models import load_model

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x, y = load_csv.load_data(False)

x_test = process_data.get_clean_data(x)
x_test = x_test.drop(['Survived'], axis=1)

print('x_test.shape: ', x_test.shape)
print('x_test.columns => \n', x_test.columns.values)
print('y.shape: ', y.shape)

model = load_model('mlp_train_model.h5')
scores = model.evaluate(x_test.values, y.values)
print('MLP, test score: ', scores[1])

with open('mlp_predict_info.txt', 'w') as file:
    file.write('test accuracy = {}\n'.format(scores[1]))
