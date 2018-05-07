import time
from common import process_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x, y = load_csv.load_data(True)

x_train = process_data.get_clean_data(x)
x_train = x_train.drop(['Survived'], axis=1)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

x_train = x_train.values
y = y.values

start = time.time()
# sgd = SGDClassifier(loss='hinge', verbose=2, alpha=0.1)
# n_batch = 20
# n_loop = len(x_train)//n_batch
# n_last_batch = len(x_train)%n_batch
# for i in range(1, n_loop+1):
#     sgd.partial_fit(x_train[(i-1)*n_batch:i*n_batch, :], y[(i-1)*n_batch:i*n_batch], classes=np.arange(0, 10))
# if n_last_batch!=0:
#     sgd.partial_fit(x_train[n_batch*n_loop:, :], y[n_batch*n_loop:])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'loss': ['hinge'], 'alpha': param_range}]

sgd = SGDClassifier(loss='hinge', verbose=2, alpha=0.1, max_iter=1000)
# sgd.fit(x_train, y)
kfold = StratifiedKFold(n_splits=5)
gs = GridSearchCV(estimator=sgd,
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=kfold)
                            # cv=5)
gs.fit(x_train, y)
train_score = 'SGD with SVM at GridSearch, train score: {}'.format(gs.best_score_)
best_param = 'SGD with SVM at GridSearch, train best param: {}'.format(gs.best_params_)
print(best_param)

# scores = cross_val_score(estimator=sgd,
#                          X=x_train,
#                          y=y,
#                          cv=kfold,
#                          # cv=5,
#                          n_jobs=1)
# train_score = 'SGD with SVM, CV train accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
print(train_score)

end = time.time()
elapsed_train_time = 'SGD with SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                   int((end - start) % 60))
print(elapsed_train_time)
