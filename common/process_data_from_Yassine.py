import pandas as pd
import numpy as np
from collections import Counter
import math


class ProcessData:
    train_data_ratio = 0.9

    def __init__(self):
        self.df_csv_train = pd.read_csv("../train.csv") #891 samples
        self.df_csv_test = pd.read_csv("../test.csv") #418 samples
        self.dataset = None

        self.df_model_train = None
        self.df_model_validation = None
        self.df_model_test = None

    @staticmethod
    def drop_outliers(df, n, features):
        """
        Takes a dataframe df of features and returns a list of the indices
        corresponding to the observations containing more than n outliers according
        to the Tukey method.
        """
        outlier_indices = []

        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        df = df.drop(multiple_outliers, axis=0)

        # return multiple_outliers

    def feature_engineering(self):
        # Join self.train and test datasets in order to obtain the same number of features during categorical conversion
        self.dataset = pd.concat(objs=[self.df_csv_train, self.df_csv_test], axis=0).reset_index(drop=True)

        # Fill Fare missing values with the median value
        self.dataset["Fare"] = self.dataset["Fare"].fillna(self.dataset["Fare"].median())

        # Apply log to Fare to reduce skewness distribution
        self.dataset["Fare"] = self.dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

        # Fill Embarked nan values of self.dataset set with 'S' most frequent value
        self.dataset["Embarked"] = self.dataset["Embarked"].fillna("S")

        # convert Sex into categorical value 0 for male and 1 for female
        self.dataset["Sex"] = self.dataset["Sex"].map({"male": 0, "female": 1})

        index_NaN_age = list(self.dataset["Age"][self.dataset["Age"].isnull()].index)

        for i in index_NaN_age:
            age_med = self.dataset["Age"].median()
            age_pred = self.dataset["Age"][(
                    (self.dataset['SibSp'] == self.dataset.iloc[i]["SibSp"]) & (self.dataset['Parch'] == self.dataset.iloc[i]["Parch"]) & (
                    self.dataset['Pclass'] == self.dataset.iloc[i]["Pclass"]))].median()
            if not np.isnan(age_pred):
                self.dataset['Age'].iloc[i] = age_pred
            else:
                self.dataset['Age'].iloc[i] = age_med

        # Get Title from Name
        self.dataset_title = [i.split(",")[1].split(".")[0].strip() for i in self.dataset["Name"]]
        self.dataset["Title"] = pd.Series(self.dataset_title)

        # Convert to categorical values Title
        self.dataset["Title"] = self.dataset["Title"].replace(
            ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
            'Rare')
        self.dataset["Title"] = self.dataset["Title"].map(
            {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
        self.dataset["Title"] = self.dataset["Title"].astype(int)

        # Drop Name variable
        self.dataset.drop(labels=["Name"], axis=1, inplace=True)

        # Create a family size descriptor from SibSp and Parch
        self.dataset["Fsize"] = self.dataset["SibSp"] + self.dataset["Parch"] + 1

        # Create new feature of family size
        self.dataset['Single'] = self.dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
        self.dataset['SmallF'] = self.dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
        self.dataset['MedF'] = self.dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
        self.dataset['LargeF'] = self.dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

        # convert to indicator values Title and Embarked
        self.dataset = pd.get_dummies(self.dataset, columns=["Title"])
        self.dataset = pd.get_dummies(self.dataset, columns=["Embarked"], prefix="Em")

        # Replace the Cabin number by the type of cabin 'X' if not
        self.dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in self.dataset['Cabin']])

        self.dataset = pd.get_dummies(self.dataset, columns=["Cabin"], prefix="Cabin")

        # Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
        Ticket = []
        for i in list(self.dataset.Ticket):
            if not i.isdigit():
                Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
            else:
                Ticket.append("X")
        self.dataset["Ticket"] = Ticket

        self.dataset = pd.get_dummies(self.dataset, columns=["Ticket"], prefix="T")

        # Create categorical values for Pclass
        self.dataset["Pclass"] = self.dataset["Pclass"].astype("category")
        self.dataset = pd.get_dummies(self.dataset, columns=["Pclass"], prefix="Pc")

        # Drop useless variables
        self.dataset.drop(labels=["PassengerId"], axis=1, inplace=True)

        self.split_data_to_train_validation()

        # detect outliers from Age, SibSp , Parch and Fare
        self.drop_outliers(self.df_model_train, 2, ["Age", "SibSp", "Parch", "Fare"])

        self.df_model_train["Survived"] = self.df_model_train["Survived"].astype(int)
        self.df_model_validation["Survived"] = self.df_model_validation["Survived"].astype(int)

    def split_data_to_train_validation(self):
        df_model_train_len = math.ceil(len(self.df_csv_train) * ProcessData.train_data_ratio)
        self.df_model_train = self.dataset.iloc[:df_model_train_len]
        self.df_model_validation = self.dataset.iloc[df_model_train_len:len(self.df_csv_train)]

    def get_train_data(self):
        return self.df_model_train

    def get_validation_data(self):
        return self.df_model_validation

    def get_test_data(self):
        return self.dataset.iloc[len(self.df_csv_train):]
