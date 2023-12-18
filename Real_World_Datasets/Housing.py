import pandas as pd
import re
import numpy as np

import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from scipy.special import kl_div
from scipy.sparse import csr_matrix
# import torch
# import torchvision
# import torchvision.datasets as datasets

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import copy


def check_certain_model(X_train, y_train):
    res = True
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    # Convert X_train and y_train to NumPy arrays
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) else y_train

    # Find indices of columns with missing values in X_train
    missing_columns_indices = np.where(pd.DataFrame(X_train).isnull().any(axis=0))[0]

    # Find rows with missing values in X_train
    missing_rows_indices = np.where(pd.DataFrame(X_train).isnull().any(axis=1))[0]

    # Record the rows with missing values and their corresponding y_train values
    X_train_missing_rows = X_train[missing_rows_indices]
    y_train_missing_rows = y_train[missing_rows_indices]

    # Remove rows with missing values from X_train and corresponding labels from y_train
    X_train_complete = np.delete(X_train, missing_rows_indices, axis=0)
    y_train_complete = np.delete(y_train, missing_rows_indices, axis=0)
    # print(X_train_complete.shape)
    # Create and train the SVM model using SGDClassifier
    svm_model = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)

    # Train the model on the data without missing values
    svm_model.fit(X_train_complete, y_train_complete)

    # Extract the feature weights (model parameters)
    feature_weights = svm_model.coef_[0]

    # Check if the absolute value of feature_weights[i] is small enough for all i with missing columns
    for i in missing_columns_indices:
        if abs(feature_weights[i]) >= 1e-3:
            res = False
            print("weight", feature_weights[i])
            break
            # Return False as soon as a condition is not met

    # Check the condition for all rows in X_train_missing_rows
    for i in range(len(X_train_missing_rows)):
        row = X_train_missing_rows[i]
        label = y_train_missing_rows[i]
        dot_product = np.sum(row[~np.isnan(row)] * feature_weights[~np.isnan(row)])
        if label * dot_product <= 1:
            print("dot product", label * dot_product)
            res = False
            break
            # Return False if the condition is not met for any row

    # If all conditions are met, return True
    return res, feature_weights

    def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns


if __name__ == '__main__':
    # df = np.load("data/CIFAR10C/snow.npy")
    # labels=np.load('data/CIFAR10C/labels.npy')
    df = pd.read_csv("data/water_potability.csv")


    # # Replace 'nan' with actual NaN values
    # df.replace('nan', np.nan, inplace=True)
    #
    # # Convert categorical columns to dummy variables
    # categorical_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    # df_categ = pd.get_dummies(df, columns=categorical_cols)
    #
    # # Reset index and update the original DataFrame
    # df = df_categ.reset_index(drop=True)

    # Extract features and target
    # Split the DataFrame into features (X) and labels (y)
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]  # Labels (the last column)

    # Split the data into training and testing sets (adjust the test_size as needed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merge X_train and y_train into df_train
    df_train = pd.concat([X_train, Y_train], axis=1)

    # Merge X_test and y_test into df_test
    df_test = pd.concat([X_test, Y_test], axis=1)

    # Reset the index for df_train and df_test
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    X_train = df_train.iloc[:, :-1]
    Y_train = df_train.iloc[:, -1]
    check_certain_model(X_train, Y_train)


