import time
from common import process_data
from common import process_train_test_data
from common import process_data_from_Yassine
import pandas as pd
from sklearn.externals import joblib
from common import load_csv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

# x, y = load_csv.load_data(True)
# x_train = process_data.get_clean_data(x)
# x_train = x_train.drop(['Survived'], axis=1)

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

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, bootstrap=False)

start = time.time()
kfold = StratifiedKFold(n_splits=10)
# param_max_depth = [4, 6, 8]
param_max_depth = [None]
param_min_samples_split = [2, 3, 10]
param_min_samples_leaf = [1, 3, 10]
param_max_features = [1, 3, 10]
param_n_estimators = [100, 300]
param_grid = {"max_depth": param_max_depth,
              "max_features": param_max_features,
              "min_samples_split": param_min_samples_split,
              "min_samples_leaf": param_min_samples_leaf,
              "n_estimators": param_n_estimators,
              }
gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold)
# cv=5)
gs.fit(x_train, y)
# clf.fit(x_train, y)
# scores = cross_val_score(estimator=clf,
#                          X=x_train,
#                          y=y,
#                          cv=kfold,
#                          # cv=5,
#                          n_jobs=1)
end = time.time()
elapsed_train_time = 'Random Forest, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                    int((end - start) % 60))
print(elapsed_train_time)
print('--------------------------------------------')
print(gs.best_estimator_)
print('--------------------------------------------')
# print('Random Forest, CV train accuracy: %s' % scores)
# train_score = 'Random Forest, CV train accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
best_score = 'Random Forest, train best score: {}'.format(gs.best_score_)
print(best_score)
best_param = 'Random Forest, train best param: {}'.format(gs.best_params_)
print(best_param)

joblib.dump(gs.best_estimator_, 'random_forest_dump.pkl')
with open('random_forest_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write('--------------------------------------------\n')
    file.write(repr(gs.best_estimator_) + '\n')
    file.write('--------------------------------------------\n')
    file.write(best_param + '\n')
    file.write(best_score + '\n')
