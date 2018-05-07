import time
from common import process_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x, y = load_csv.load_data(True)

x_train = process_data.get_clean_data(x)
x_train = x_train.drop(['Survived'], axis=1)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, max_depth=4)
# pipe_rf = Pipeline([('scl', StandardScaler()),
#                     ('clf', RandomForestClassifier(n_estimators=10000,
#                                                    random_state=0,
#                                                    n_jobs=-1))])

start = time.time()
kfold = StratifiedKFold(n_splits=10)
param_grid = [{'max_depth': [4, 6, 8]}]
# param_grid = [{'max_depth': [4, 6, 8],
#                # "max_features": [1, 3, 9],
#                "min_samples_split": [2, 3, 10],
#                "min_samples_leaf": [1, 3, 10]}]
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
# print('Random Forest, CV train accuracy: %s' % scores)
# train_score = 'Random Forest, CV train accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
best_score = 'Random Forest, train best score: {}'.format(gs.best_score_)
best_param = 'Random Forest, train best param: {}'.format(gs.best_params_)
print(best_param)
print(best_score)
joblib.dump(gs.best_estimator_, 'random_forest_dump.pkl')
with open('random_forest_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write(best_param + '\n')
    file.write(best_score + '\n')
