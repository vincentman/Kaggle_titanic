import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

train_data_ratio = 0.7


def get_train_data(x):
    return x.loc[:math.ceil(891 * train_data_ratio)]


def get_validation_data(x):
    return x.iloc[math.ceil(891 * train_data_ratio):891]


def get_test_data(x):
    return x.iloc[891:]


def get_clean_data():
    # 891 samples
    pd_train_csv = pd.read_csv('../train.csv')
    # 418 samples
    pd_test_csv = pd.read_csv('../test.csv')
    # 1309 samples
    x = pd.concat([pd_train_csv, pd_test_csv])

    print('Before cleaning data, the number of column value is NaN =\n', x.isnull().sum())

    # 1. fill NaN for Age
    # print(x['Age'].isnull().sum())
    # print(x[x['Age'].isnull()])
    # 1.1. 將 Ms.,Miss., Mrs. 缺值的 Age，以其中位數取代
    mask = (x['Age'].isnull()) & (
                (x['Name'].str.contains('Ms.')) | (x['Name'].str.contains('Miss.')) | (x['Name'].str.contains('Mrs.')))
    mask2 = ((x['Name'].str.contains('Ms.')) | (x['Name'].str.contains('Miss.')) | (x['Name'].str.contains('Mrs.')))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x.loc[mask2, 'Age'].median())
    # 1.2. 將 Mr.,Sir.,Major 缺值的 Age，以其中位數取代
    mask = (x['Age'].isnull()) & (
            (x['Name'].str.contains('Mr.')) | (x['Name'].str.contains('Sir.')) | (x['Name'].str.contains('Major')))
    mask2 = ((x['Name'].str.contains('Mr.')) | (x['Name'].str.contains('Sir.')) | (x['Name'].str.contains('Major')))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x.loc[mask2, 'Age'].median())
    # 1.3. 將 Master. 缺值的 Age，以其中位數取代
    mask = (x['Age'].isnull()) & (x['Name'].str.contains('Master.'))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x[x['Name'].str.contains('Master.')]['Age'].median())
    # 1.4. 將 Dr. 缺值的 Age，以其中位數取代
    mask = (x['Age'].isnull()) & (x['Name'].str.contains('Dr.'))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x[x['Name'].str.contains('Dr.')]['Age'].median())
    # print("After filling, the number of 'Age' value is NaN = ", x['Age'].isnull().sum())

    # 2. fill NaN for Embarked
    # print(x[x['Embarked'].isnull()])
    # print(x['Embarked'].isnull().sum())
    x['Embarked'].fillna("C", inplace=True)
    # print("After filling, the number of 'Embarked' value is NaN = ", x['Embarked'].isnull().sum())

    # 3. combine SibSp and Parch into Family
    x['Family'] = x['SibSp'] + x['Parch']

    # 4. Extract Name into Title1
    x['Title1'] = x['Name'].str.split(", ", expand=True)[1]
    x['Title1'] = x['Title1'].str.split(".", expand=True)[0]
    print('Average Age:\n', x.groupby(['Title1'])['Age'].mean())
    print('=======================')
    print('count:\n', x.groupby(['Title1', 'Sex'])['Name'].count())
    print('=======================')
    print('Average Age:\n', x.groupby(['Title1', 'Pclass'])['Age'].mean())
    print('=======================')

    # 5. convert Title1 into Title2
    x['Title2'] = x['Title1'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'the Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don',
         'Dona'],
        ['Miss', 'Mrs', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mrs'])
    print('count:\n', x.groupby(['Title2', 'Sex'])['Name'].count())
    print('=======================')
    print('Average Age:\n', x.groupby(['Title2', 'Pclass'])['Age'].mean())
    print('=======================')

    # 6. Extract Ticket into Ticket_info
    x['Ticket_info'] = x['Ticket'].apply(
        lambda x: x.replace(".", "").replace("/", "").strip().split(' ')[0] if not x.isdigit() else 'X')
    # print('count:\n', x.groupby(['Ticket_info', 'Survived'])['Name'].count())

    # 7. Cabin：取出最前面的英文字母，缺值的用'NoCabin'來取代
    x["Cabin"] = x['Cabin'].apply(lambda x: str(x)[0] if not pd.isnull(x) else 'NoCabin')
    # print('count:\n', x.groupby(['Cabin', 'Survived'])['Name'].count())

    # 8. fill NaN for Fare: mean
    x['Fare'] = x['Fare'].fillna(x['Fare'].mean())

    #  drop unused columns
    x = x.drop(['Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Title1'], axis=1)

    print('After cleaning data, the number of column value is NaN =\n', x.isnull().sum())
    print('=======================')
    # print(x.describe())
    # print(x.describe(include=['O']))

    # encode data as digit
    # one-hot encoding
    # x = pd.get_dummies(data=x, columns=['Embarked'])
    x['Embarked'] = x['Embarked'].astype('category').cat.codes
    # x['Sex'] = x['Sex'].map({'female': 0, 'male': 1}).astype(int)
    x['Sex'] = x['Sex'].astype('category').cat.codes
    # x['Title1'] = x['Title1'].astype('category').cat.codes
    x['Title2'] = x['Title2'].astype('category').cat.codes
    x['Cabin'] = x['Cabin'].astype('category').cat.codes
    x['Ticket_info'] = x['Ticket_info'].astype('category').cat.codes

    return x


def get_feature_importance(column_names, x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import matplotlib.pyplot as plt
    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                column_names[f],
                                importances[indices[f]]))

    # plt.title('Feature Importances')
    # plt.bar(range(x_train.shape[1]),
    #         importances[indices],
    #         color='lightblue',
    #         align='center')
    #
    # plt.xticks(range(x_train.shape[1]), x_train, rotation=90)
    # plt.xlim([-1, x_train.shape[1]])
    # plt.tight_layout()
    # # plt.savefig('./figures/random_forest.png', dpi=300)
    # plt.show()
