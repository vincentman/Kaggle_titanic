import pandas as pd

def get_clean_data(x):
    print('Before cleaning data, the number of column value is NaN = ', x.isnull().sum())

    # 1. fill NaN for Age
    # print(x['Age'].isnull().sum())
    # print(x[x['Age'].isnull()])
    # 1.1. 將 Ms.,Miss. 缺值的 Age，以其中位數取代
    mask = (x['Age'].isnull()) & ((x['Name'].str.contains('Ms.')) | (x['Name'].str.contains('Miss.')))
    mask2 = ((x['Name'].str.contains('Ms.')) | (x['Name'].str.contains('Miss.')))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x.loc[mask2, 'Age'].median())
    # 1.2. 將 Mr.,Sir.,Major 缺值的 Age，以其中位數取代
    mask = (x['Age'].isnull()) & (
            (x['Name'].str.contains('Mr.')) | (x['Name'].str.contains('Sir.')) | (x['Name'].str.contains('Major')))
    mask2 = ((x['Name'].str.contains('Mr.')) | (x['Name'].str.contains('Sir.')) | (x['Name'].str.contains('Major')))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x.loc[mask2, 'Age'].median())

    mask = (x['Age'].isnull()) & (x['Name'].str.contains('Master.'))
    x.loc[mask, 'Age'] = x.loc[mask, 'Age'].fillna(x[x['Name'].str.contains('Master.')]['Age'].median())
    # print("After filling, the number of 'Age' value is NaN = ", x['Age'].isnull().sum())

    # 2. fill NaN for Embarked
    # print(x[x['Embarked'].isnull()])
    # print(x['Embarked'].isnull().sum())
    x['Embarked'].fillna("C", inplace=True)
    # print("After filling, the number of 'Embarked' value is NaN = ", x['Embarked'].isnull().sum())

    print('After cleaning data, the number of column value is NaN = ', x.isnull().sum())

    # 3. drop unrelated columns
    x = x.drop(['Cabin', 'Name', 'Ticket', 'Survived', 'PassengerId'], axis=1)

    # print(x.describe())
    # print(x.describe(include=['O']))

    # encode 'Embarked' as digit
    # x['Embarked'] = x['Embarked'].astype('category').cat.codes
    # one-hot encoding
    x = pd.get_dummies(data=x,columns=['Embarked'])
    x['Sex'] = x['Sex'].map({'female': 0, 'male': 1}).astype(int)

    return x
