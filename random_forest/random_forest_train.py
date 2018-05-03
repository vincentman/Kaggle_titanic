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

# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)

x, y = load_csv.load_data(True)

x_train = process_data.get_clean_data(x)
x_train = x_train.drop(['Survived'], axis=1)

print('x_train.shape: ', x_train.shape)
print('x_train.columns => \n', x_train.columns.values)
print('y.shape: ', y.shape)

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
pipe_rf = Pipeline([
    ('clf', clf)])
# pipe_rf = Pipeline([('scl', StandardScaler()),
#                     ('clf', RandomForestClassifier(n_estimators=10000,
#                                                    random_state=0,
#                                                    n_jobs=-1))])

start = time.time()
scores = cross_val_score(estimator=pipe_rf,
                         X=x_train,
                         y=y,
                         cv=5,
                         n_jobs=1)
clf.fit(x_train, y)
end = time.time()
elapsed_train_time = 'Random Forest, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                          int((end - start) % 60))
print(elapsed_train_time)
# print('Random Forest, CV train accuracy: %s' % scores)
train_score = 'Random Forest, CV train accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
# train_score = 'Random Forest, train accuracy: {}'.format(clf.score(x_train, y))
print(train_score)
joblib.dump(clf, 'random_forest_dump.pkl')
with open('random_forest_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write('train accuracy = {}'.format(train_score))
