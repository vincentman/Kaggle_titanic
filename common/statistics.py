import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def show_statistics(x):
    # print(x.head())
    # print(x.describe())
    # print(x.info())
    # print(x.describe(include=['O']))

    # fill NaN for Age
    x.loc[x['Age'].isnull(), 'Age'] = x['Age'].mean()
    # x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].mean())

    # 各年齡層的男女人數：圖形化
    fig = plt.figure(figsize=(15, 8))
    plt.hist([x[x['Sex'] == 'male']['Age'], x[x['Sex'] == 'female']['Age']], stacked=False, color=['g', 'r'], bins=30,
             label=['Male', 'Female'])
    plt.xlabel('Age')
    plt.ylabel('Number of Sex')
    plt.legend()

    # 以人數比例來看不同性別的存活率
    print(x[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    # 圖形化
    survived_sex = x[x['Survived'] == 1]['Sex'].value_counts()
    dead_sex = x[x['Survived'] == 0]['Sex'].value_counts()
    df = pd.DataFrame([survived_sex, dead_sex])
    df.index = ['Survived', 'Dead']
    # df.plot(kind='bar', stacked=True, figsize=(15, 8))

    # 以人數來看不同性別的存活率：圖形化
    total_sex = x['Sex'].value_counts()
    p_survived_sex = x[x['Survived'] == 1]['Sex'].value_counts() / total_sex
    p_dead_sex = x[x['Survived'] == 0]['Sex'].value_counts() / total_sex
    df = pd.DataFrame([p_survived_sex, p_dead_sex])
    df.index = ['Survived', 'Dead']
    # df.plot(kind='bar', stacked=True, figsize=(15, 8))

    # 統計姓名裡的稱謂總人數與其年齡的平均數
    for title in ['Mr.', 'Sir.', 'Dr.', 'Major.', 'Master.']:
        num = x[(x['Name'].str.contains(title))]['Name'].count()
        age = x[(x['Name'].str.contains(title))]['Age'].mean()
        print('{} – > {} males, Age average is {}'.format(title, num, age))
    print('-----------------------')
    for title in ['Ms.', 'Miss.', 'Mrs.', 'Lady.']:
        num = x[(x['Name'].str.contains(title))]['Name'].count()
        age = x[(x['Name'].str.contains(title))]['Age'].mean()
        print('{} – > {} females, Age average is {}'.format(title, num, age))
    print('=======================')

    # 統計某稱謂的總人數、存活人數、死亡人數、存活率
    for title in ['Mr.', 'Sir.', 'Dr.', 'Major.', 'Master.']:
        num_survived = x[(x['Survived'] == 1) & (x['Name'].str.contains(title))]['Name'].count()
        num_died = x[(x['Survived'] == 0) & (x['Name'].str.contains(title))]['Name'].count()
        print('{} total:{} – > {} survived, {} died. {:.3f}% survived'.format(title, num_survived + num_died,
                                                                              num_survived,
                                                                              num_died, (100 * num_survived / (
                    num_survived + num_died))))
    print('-----------------------')
    for title in ['Ms.', 'Miss.', 'Mrs.', 'Lady.']:
        num_survived = x[(x['Survived'] == 1) & (x['Name'].str.contains(title))]['Name'].count()
        num_died = x[(x['Survived'] == 0) & (x['Name'].str.contains(title))]['Name'].count()
        print('{} total:{} – > {} survived, {} died. {:.3f}% survived'.format(title, num_survived + num_died,
                                                                              num_survived,
                                                                              num_died, (100 * num_survived / (
                    num_survived + num_died))))
    print('=======================')

    # 從人數來看不同船票等級的存活率：圖形化
    survived_pclass = x[x['Survived'] == 1]['Pclass'].value_counts()
    dead_pclass = x[x['Survived'] == 0]['Pclass'].value_counts()
    df = pd.DataFrame([survived_pclass, dead_pclass])
    df.index = ['Survived', 'Dead']
    # df.plot(kind='bar', stacked=False, figsize=(15, 8))

    # 從人數比例來看不同船票等級的存活率
    print(x[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                           ascending=False))
    print('=======================')

    # 不同船票等級與性別對於存活率的影響
    print(x[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))
    print('=======================')

    # 不同船票等級與性別對於存活率的影響：圖形化(女性)
    total_female_p1 = x[(x['Pclass'] == 1) & (x['Sex'] == "female")]['Survived'].count()
    female_p1 = x[(x['Pclass'] == 1) & (x['Sex'] == "female")]['Survived'].value_counts() / total_female_p1
    total_female_p2 = x[(x['Pclass'] == 2) & (x['Sex'] == "female")]['Survived'].count()
    female_p2 = x[(x['Pclass'] == 2) & (x['Sex'] == "female")]['Survived'].value_counts() / total_female_p2
    total_female_p3 = x[(x['Pclass'] == 3) & (x['Sex'] == "female")]['Survived'].count()
    female_p3 = x[(x['Pclass'] == 3) & (x['Sex'] == "female")]['Survived'].value_counts() / total_female_p3
    df = pd.DataFrame([female_p1[[0, 1]], female_p2[[0, 1]], female_p3[[0, 1]]])
    df.index = ['Female in P1', 'Female in P2', 'Female in P3']
    df.plot(kind='bar', stacked=False, figsize=(15, 8))

    # 不同年齡層的存亡人數：圖形化
    plt.figure(figsize=(15, 8))
    plt.hist([x[x['Survived'] == 1]['Age'], x[x['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'], bins=30,
             label=['Survived', 'Dead'])
    plt.xlabel('Age')
    plt.ylabel('Number of passengers')
    plt.legend()

    # 親屬人數對存活率的影響
    x_with_family = x.copy()
    x_with_family['Family'] = x['SibSp'] + x['Parch']
    print(x_with_family[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))
    # print('=======================')

    # 上岸港口對存活率的影響
    x[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    # print('=======================')

    # 上岸港口對存活率的影響：圖形化
    total_Embarked_S = x[x['Embarked'] == 'S']['Survived'].count()
    total_Embarked_C = x[x['Embarked'] == 'C']['Survived'].count()
    total_Embarked_Q = x[x['Embarked'] == 'Q']['Survived'].count()
    Embarked_S = x[x['Embarked'] == 'S']['Survived'].value_counts() / total_Embarked_S
    Embarked_C = x[x['Embarked'] == 'C']['Survived'].value_counts() / total_Embarked_C
    Embarked_Q = x[x['Embarked'] == 'Q']['Survived'].value_counts() / total_Embarked_Q
    df = pd.DataFrame([Embarked_S, Embarked_C, Embarked_Q])
    df.index = ['Southampton', 'Cherbourg', 'Queenstown']
    # df.plot(kind='bar', stacked = False, figsize = (15, 8))

    # 上岸港口與船票等級的關係
    total_Pclass_S = x[x['Embarked'] == 'S']['Pclass'].count()
    total_Pclass_C = x[x['Embarked'] == 'C']['Pclass'].count()
    total_Pclass_Q = x[x['Embarked'] == 'Q']['Pclass'].count()
    Embarked_S = x[x['Embarked'] == 'S']['Pclass'].value_counts() / total_Pclass_S
    Embarked_C = x[x['Embarked'] == 'C']['Pclass'].value_counts() / total_Pclass_C
    Embarked_Q = x[x['Embarked'] == 'Q']['Pclass'].value_counts() / total_Pclass_Q
    df = pd.DataFrame([Embarked_S, Embarked_C, Embarked_Q])
    df.index = ['Southampton', 'Cherbourg', 'Queenstown']
    # df.plot(kind='bar', stacked=False, figsize=(15, 8))

    # plt.show()


# used for MLP
def show_train_history(train_history, train_acc, validation_acc, ylabel):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[validation_acc])
    epoch_num = len(train_history.epoch)
    final_epoch_train_acc = train_history.history[train_acc][epoch_num - 1]
    final_epoch_validation_acc = train_history.history[validation_acc][epoch_num - 1]
    plt.text(epoch_num, final_epoch_train_acc, 'train = {:.3f}'.format(final_epoch_train_acc))
    plt.text(epoch_num, final_epoch_validation_acc-0.01, 'valid = {:.3f}'.format(final_epoch_validation_acc))
    plt.title('Train History')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.xlim(xmax=epoch_num+1)
    plt.legend(['train', 'validation'], loc='upper left')
    fig = plt.gcf()
    fig.savefig('./mlp_train_{}.png'.format(ylabel), dpi=100)
    plt.clf()
    # plt.show()
    return final_epoch_train_acc, final_epoch_validation_acc
