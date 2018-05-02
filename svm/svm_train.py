import time
from sklearn.svm import SVC
from common import process_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x, y = load_csv.load_data(True)

x_train = process_data.get_clean_data(x)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.values)
y_train = y.values

start = time.time()
# param_range = [0.01, 0.1, 1.0, 10.0]
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}]
svm = SVC(random_state=0, verbose=False)
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
gs.fit(x_train, y_train)
print('\nSVM at GridSearch, best score: ', gs.best_score_)
print('SVM at GridSearch, train best param: ', gs.best_params_)
end = time.time()
print('SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60), int((end - start) % 60)))
best_clf = gs.best_estimator_
best_clf.fit(x_train, y_train)
print('SVM, train accuracy: ', best_clf.score(x_train, y_train))
joblib.dump(best_clf, 'svm_dump.pkl')




