#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time
import unicodedata
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable the warning
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, vstack
import random
import pickle
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from tabulate import tabulate
from sklearn import svm
from sklearn.impute import KNNImputer

# In[2]:




def check_certain_model(X, y):


    # Convert the numpy array to a DataFrame
    df_train = pd.DataFrame(np.column_stack((X, y)))

    # Split df_train into df_train_complete (rows without missing data) and df_train_missing (rows with missing values)
    df_train_complete = df_train.dropna()
    df_train_missing = df_train[~df_train.index.isin(df_train_complete.index)]

    # Reset index for all dataframes
    df_train.reset_index(drop=True, inplace=True)
    df_train_complete.reset_index(drop=True, inplace=True)
    df_train_missing.reset_index(drop=True, inplace=True)

    X_train_complete = df_train_complete.iloc[:, :-1].values
    y_train_complete = df_train_complete.iloc[:, -1].values

    # Create and train a Polynomial Kernel SVM classifier
    # You can adjust the degree and other hyperparameters as needed
    poly_svm = SVC(kernel='poly', degree=3, C=1)
    poly_svm.fit(X_train_complete, y_train_complete)

    # Print the support vectors
    support_vectors = poly_svm.support_vectors_

    missing_columns_indices = df_train_missing.columns[df_train_missing.isna().any()].tolist()

    res = True

    for sv in support_vectors:
        for index in missing_columns_indices:
            if sv[index] != 0:
                res = False
                break
        if not res:
            break

    df_train_missing.fillna(0, inplace=True)

    features = df_train_missing.iloc[:, :-1]  # All columns except the last one
    labels = df_train_missing.iloc[:, -1]  # The last column (labels)

    for index, example in features.iterrows():
        predicted_class = poly_svm.predict([example])  # Predict the class for the features
        decision_function_value = poly_svm.decision_function([example])  # Calculate the decision function value

        true_class_label = labels.loc[index]  # Get the true class label for this example

        if predicted_class == true_class_label:
            if abs(decision_function_value) <= 1:
                print(abs(decision_function_value))
                # If any example is correctly classified but within the margin, set res to False
                res = False
                #break  # Break out of the loop as soon as res becomes False
        else:
            # If any example is misclassified, set res to False
            res = False
            print("misclassified")
            break  # Break out of the loop as soon as res becomes False

    return res



# In[6]:
df = pd.read_csv('HMEQ-processed.csv')
# Check for missing values in each column
missing_values_per_column = df.isnull().sum()

# Count the number of columns with missing values
columns_with_missing_values = missing_values_per_column[missing_values_per_column > 0].index
num_columns_with_missing_values = len(columns_with_missing_values)

# Print the result
print(f"Number of columns with missing values: {num_columns_with_missing_values}")
print(f"Columns with missing values: {columns_with_missing_values.tolist()}")


X = df.drop(columns=['Happy with online education?_Yes'])
y = df['Happy with online education?_Yes']
y.loc[y == 0] = -1
y = y.reset_index(drop=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n, d = X_train.shape  # Get the number of samples and features from the feature matrix
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
X_test_clean = X_test.dropna()
y_test_clean = y_test[X_test_clean.index]
X_test_clean.reset_index(drop=True, inplace=True)
y_test_clean.reset_index(drop=True, inplace=True)
print(np.shape(X_test_clean))


start_time = time.time()
CM_result = check_certain_model(X_train, y_train)
end_time = time.time()
CM_time = end_time - start_time
    
print("CM time", CM_time)
print("CM result", CM_result)

# Perform Mean Imputation and train the model
start_time = time.time()
X_train_mean = X_train.fillna(X_train.mean())
X_test_mean = X_test.fillna(X_train.mean())
svm_model_mean = SVC(kernel='poly', degree=3, C=1)
svm_model_mean.fit(X_train_mean, y_train)
accuracy_mean = svm_model_mean.score(X_test_mean, y_test)
end_time = time.time()
elapsed_time_mean = end_time - start_time

# Perform Deletion and train the model
start_time = time.time()
X_train_delete = X_train.dropna()
y_train_delete = y_train[X_train.index.isin(X_train_delete.index)]
X_test_delete = X_test.dropna()
y_test_delete = y_test[X_test.index.isin(X_test_delete.index)]
svm_model_delete = SVC(kernel='poly', degree=3, C=1)
svm_model_delete.fit(X_train_delete, y_train_delete)
accuracy_delete = svm_model_delete.score(X_test_delete, y_test_delete)
end_time = time.time()
elapsed_time_delete = end_time - start_time

# Perform KNN Imputation and train the model
start_time = time.time()
imputer = KNNImputer(n_neighbors=5)
X_train_knn = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_knn = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
svm_model_knn = SVC(kernel='poly', degree=3, C=1)
svm_model_knn.fit(X_train_knn, y_train)
accuracy_knn = svm_model_knn.score(X_test_knn, y_test)
end_time = time.time()
elapsed_time_knn = end_time - start_time

