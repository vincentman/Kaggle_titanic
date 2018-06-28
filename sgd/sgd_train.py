import time
from common import process_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from common import load_csv
from common import process_data_from_Yassine
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

process_data = process_data_from_Yassine.ProcessData()
process_data.feature_engineering()
train_data = process_data.get_train_data()
y = train_data.Survived
x_train = train_data.drop(['Survived'], axis=1)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

x_train = StandardScaler().fit_transform(x_train.values)

start = time.time()
kfold = StratifiedKFold(n_splits=5)

# ----------- use partial_fit -----------
sgd = SGDClassifier(loss='hinge', verbose=2, alpha=0.0001, penalty='elasticnet', tol=1e-3, l1_ratio=0.2)
n_batch = 20
n_loop = len(x_train) // n_batch
n_last_batch = len(x_train) % n_batch
for i in range(1, n_loop + 1):
    sgd.partial_fit(x_train[(i - 1) * n_batch:i * n_batch, :], y[(i - 1) * n_batch:i * n_batch],
                    classes=[0, 1])
if n_last_batch != 0:
    sgd.partial_fit(x_train[n_batch * n_loop:, :], y[n_batch * n_loop:])

# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_l1_ratio = np.arange(0.1, 1, 0.1)
# param_grid = {'loss': ['hinge'], 'alpha': param_range, 'l1_ratio': param_l1_ratio}
#
# sgd = SGDClassifier(loss='hinge', verbose=2, max_iter=None, penalty='elasticnet', tol=1e-3)
# gs = GridSearchCV(estimator=sgd,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=kfold)
# # cv=5)
# gs.fit(x_train, y)
# end = time.time()
# elapsed_train_time = 'SGD with SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
#                                                                                    int((end - start) % 60))
# print(elapsed_train_time)
# print('--------------------------------------------')
# best_clf = gs.best_estimator_
# print(best_clf)
# print('--------------------------------------------')
# best_score = 'SGD with SVM at GridSearch, train best score: {}'.format(gs.best_score_)
# print(best_score)
# best_param = 'SGD with SVM at GridSearch, train best param: {}'.format(gs.best_params_)
# print(best_param)

scores = cross_val_score(estimator=sgd,
                         X=x_train,
                         y=y,
                         cv=kfold,
                         # cv=5,
                         n_jobs=1)
train_score = 'SGD with SVM, CV train accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
print(train_score)

# joblib.dump(best_clf, 'sgd_dump.pkl')
# with open('sgd_train_info.txt', 'w') as file:
#     file.write(elapsed_train_time + '\n')
#     file.write('--------------------------------------------\n')
#     file.write(repr(best_clf) + '\n')
#     file.write('--------------------------------------------\n')
#     file.write(best_param + '\n')
#     file.write(best_score + '\n')
