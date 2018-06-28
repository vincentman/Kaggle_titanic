import time
from sklearn.svm import SVC
from common import process_train_test_data
from common import process_data_from_Yassine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# all_data = process_train_test_data.get_clean_data()
# train_data = process_train_test_data.get_train_data(all_data)

process_data = process_data_from_Yassine.ProcessData()
process_data.feature_engineering()
train_data = process_data.get_train_data()
y = train_data.Survived
x_train = train_data.drop(['Survived'], axis=1)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

x_train = StandardScaler().fit_transform(x_train.values)
y_train = y.values

start = time.time()
# param_range = [0.01, 0.1, 1.0, 10.0]
param_C = [1, 10, 50, 100, 200, 300, 1000]
param_gamma = [0.0001, 0.001, 0.01, 0.1, 1.0]
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = {'C': param_C, 'gamma': param_gamma, 'kernel': ['rbf']}
svm = SVC(random_state=0, verbose=False)

kfold = StratifiedKFold(n_splits=10)
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold)
# cv=5)
gs.fit(x_train, y_train)
end = time.time()
elapsed_train_time = 'SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                          int((end - start) % 60))
print(elapsed_train_time)
best_clf = gs.best_estimator_
print('--------------------------------------------')
print(best_clf)
print('--------------------------------------------')
best_score = 'SVM at GridSearch, train best score: {}'.format(gs.best_score_)
print(best_score)
best_param = 'SVM at GridSearch, train best param: {}'.format(gs.best_params_)
print(best_param)

joblib.dump(best_clf, 'svm_dump.pkl')
with open('svm_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write('--------------------------------------------\n')
    file.write(repr(best_clf) + '\n')
    file.write('--------------------------------------------\n')
    file.write(best_param + '\n')
    file.write(best_score + '\n')
