import time
from common import process_data
import pandas as pd
from sklearn.externals import joblib
from common import load_csv
import numpy as np
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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=None)

# clf = AdaBoostClassifier(base_estimator=tree,
#                          n_estimators=1000,
#                          learning_rate=0.1,
#                          random_state=0)
clf = AdaBoostClassifier(tree, random_state=7)

param_grid = {"base_estimator__criterion": ["gini", "entropy"],
              "base_estimator__splitter": ["best", "random"],
              "n_estimators": [1000],
              "learning_rate": [0.0001, 0.001, 0.01, 0.1]}
# param_grid = {"base_estimator__criterion": ["gini", "entropy"],
#                   "base_estimator__splitter": ["best", "random"],
#                   "algorithm": ["SAMME", "SAMME.R"],
#                   "n_estimators": [100, 200],
#                   "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

kfold = StratifiedKFold(n_splits=10)
start = time.time()
gs = GridSearchCV(clf, param_grid=param_grid, cv=kfold, scoring="accuracy", verbose=1)

gs.fit(x_train, y)
best_score = 'AdaBoost at GridSearch, best score: {}'.format(gs.best_score_)
print('\n{}', best_score)
best_param = 'AdaBoost at GridSearch, train best param: {}'.format(gs.best_params_)
print(best_param)
end = time.time()
elapsed_train_time = 'AdaBoost, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                               int((end - start) % 60))
print(elapsed_train_time)
best_clf = gs.best_estimator_
joblib.dump(best_clf, 'adaboost_dump.pkl')
with open('adaboost_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write(best_score + '\n')
    file.write(best_param + '\n')
