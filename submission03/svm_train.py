import time
from sklearn.svm import SVC
from common import process_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x, y = load_csv.load_data(True)

x_train = process_data.get_clean_data(x)
x_train = x_train.drop(['Survived'], axis=1)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

x_train = StandardScaler().fit_transform(x_train.values)
y_train = y.values

start = time.time()
# param_range = [0.01, 0.1, 1.0, 10.0]
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}]
svm = SVC(random_state=0, verbose=False)

kfold = StratifiedKFold(n_splits=10)
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold)
                  # cv=5)
gs.fit(x_train, y_train)
print('\nSVM at GridSearch, best score: ', gs.best_score_)
best_param = 'SVM at GridSearch, train best param: {}'.format(gs.best_params_)
print(best_param)
end = time.time()
elapsed_train_time = 'SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                          int((end - start) % 60))
print(elapsed_train_time)
best_clf = gs.best_estimator_
best_clf.fit(x_train, y_train)
train_score = best_clf.score(x_train, y_train)
print('SVM, train accuracy: ', train_score)
joblib.dump(best_clf, 'svm_dump.pkl')
with open('svm_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write(best_param + '\n')
    file.write('train accuracy = {}'.format(train_score))
