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


# In[2]:


def generate_mock_data(n_samples, n_missing, n_features):
    # Generate your data
    n_complete_samples = n_samples - n_missing
    random_cols = np.random.uniform(low=0, high=1, size=(n_complete_samples // 2, n_features))
    ones_col = np.ones((n_complete_samples // 2, 1))
    original_array = np.hstack((random_cols, ones_col))
    original_array[:, 0] = 0
    new_array = np.copy(original_array)
    new_array[:, 1:] *= -1
    merged_array = np.concatenate((original_array, new_array), axis=0)
    features, target = merged_array[:, :-1], merged_array[:, -1]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a polynomial kernel SVM to identify support vectors
    svm_model = SVC(kernel='poly', degree=3, C=1)
    svm_model.fit(X_train, y_train)
    
    # Create an empty list to store examples that satisfy the conditions

    outside_margin_examples = []

    # Counter for the number of satisfying examples
    satisfying_count = 0
    # Make sure the number of features in outside_margin_examples matches the number of features in merged_array
    n_features = merged_array.shape[1] - 1  # Subtract 1 to exclude the target column

    while satisfying_count < n_missing:
        random_values = np.random.uniform(low=0, high=1, size=n_features - 1)
        example = np.insert(random_values, 0, 0)
    
        # Predict the class label for the example using the svm_model
        predicted_class = svm_model.predict([example])[0]
    
        decision_function_value = svm_model.decision_function([example])[0]
    
        if abs(decision_function_value) > 1:
            label = 1 if predicted_class == 1 else -1
            outside_margin_example = np.append(example, label)  # Include the label
            outside_margin_examples.append(outside_margin_example)
            satisfying_count += 1

 


    # Convert the list of examples to a NumPy array
    outside_margin_examples = np.array(outside_margin_examples)
    data_set = np.vstack((merged_array, outside_margin_examples))
    return data_set


# In[3]:


# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# def generate_mock_data(n_samples, n_missing, n_features):
#     # Generate your data
#     n_complete_samples = n_samples - n_missing
#     random_cols = np.random.uniform(low=0, high=1, size=(n_complete_samples // 2, n_features))
#     ones_col = np.ones((n_complete_samples // 2, 1))
#     original_array = np.hstack((random_cols, ones_col))
    
#     # Randomly choose a column to set to zeros
#     random_column_index = np.random.randint(1, n_features)  # Avoid setting the target column to zeros
#     original_array[:, random_column_index] = 0
    
#     new_array = np.copy(original_array)
    
#     # Multiply all columns other than random_column_index by -1
#     for col in range(n_features):
#         if col != random_column_index:
#             new_array[:, col] *= -1
    
#     merged_array = np.concatenate((original_array, new_array), axis=0)
#     features, target = merged_array[:, :-1], merged_array[:, -1]

#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#     # Train a polynomial kernel SVM to identify support vectors
#     svm_model = SVC(kernel='poly', degree=3, C=1)
#     svm_model.fit(X_train, y_train)
    
#     # Create an empty list to store examples that satisfy the conditions
#     outside_margin_examples = []

#     # Counter for the number of satisfying examples
#     satisfying_count = 0
    
#     # Make sure the number of features in outside_margin_examples matches the number of features in merged_array
#     n_features = merged_array.shape[1] - 1  # Subtract 1 to exclude the target column

#     while satisfying_count < n_missing:
#         random_values = np.random.uniform(low=0, high=1, size=n_features - 1)
        
#         random_values = np.insert(random_values, random_column_index, 0)
        
#         # Predict the class label for the example using the svm_model
#         predicted_class = svm_model.predict([random_values])[0]
    
#         decision_function_value = svm_model.decision_function([random_values])[0]
    
#         if abs(decision_function_value) > 1:
#             label = 1 if predicted_class == 1 else -1
#             outside_margin_example = np.append(random_values, label)  # Include the label
#             outside_margin_examples.append(outside_margin_example)
#             satisfying_count += 1

#     # Convert the list of examples to a NumPy array
#     outside_margin_examples = np.array(outside_margin_examples)
#     data_set = np.vstack((merged_array, outside_margin_examples))
#     return data_set


# In[4]:


def check_certain_model(data_set):
    X = data_set[:, :-1]  # Features
    y = data_set[:, -1]   # Target

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


# In[5]:


# def check_certain_model(data_set):
#     X = data_set[:, :-1]  # Features
#     y = data_set[:, -1]   # Target

#     # Convert the numpy array to a DataFrame
#     df_train = pd.DataFrame(np.column_stack((X, y)))

#     # Split df_train into df_train_complete (rows without missing data) and df_train_missing (rows with missing values)
#     df_train_complete = df_train.dropna()
#     df_train_missing = df_train[~df_train.index.isin(df_train_complete.index)]

#     # Reset index for all dataframes
#     df_train.reset_index(drop=True, inplace=True)
#     df_train_complete.reset_index(drop=True, inplace=True)
#     df_train_missing.reset_index(drop=True, inplace=True)

#     X_train_complete = df_train_complete.iloc[:, :-1].values
#     y_train_complete = df_train_complete.iloc[:, -1].values

#     # Create and train a Polynomial Kernel SVM classifier
#     # You can adjust the degree and other hyperparameters as needed
#     poly_svm = SVC(kernel='poly', degree=3, C=1)
#     poly_svm.fit(X_train_complete, y_train_complete)

#     # Print the support vectors
#     support_vectors = poly_svm.support_vectors_

#     missing_columns_indices = df_train_missing.columns[df_train_missing.isna().any()].tolist()

#     res = True

#     for sv in support_vectors:
#         for index in missing_columns_indices:
#             if sv[index] != 0:
#                 res = False
#                 break
#         if not res:
#             break

#     df_train_missing.fillna(0, inplace=True)

#     decision_values = poly_svm.decision_function(df_train_missing.iloc[:, :-1])  # Exclude the label column

#     # Check if any new example in df_train_missing is a support vector
#     if any(np.abs(decision_values) < 1):
#         res = False
        
#     print(decision_values)
#     return res


# In[6]:


def test(n_samples, n_features, missing_factor):
    n_missing = round(n_samples*missing_factor)
    
    data_set = generate_mock_data(n_samples = n_samples,n_missing = n_missing, n_features = n_features)

    missing_columns = np.isnan(data_set).any(axis=0)
    columns_with_missing_data = np.where(missing_columns)[0]

  
    start_time = time.time()
    CM_result = check_certain_model(data_set)
    end_time = time.time()
    CM_time = end_time - start_time
    
    return n_samples, n_features, missing_factor,n_missing,CM_result, CM_time


# In[7]:


results_dict = {'n_samples': [], 'missing_factor': [], 'n_features': [], 'n_missing': [],'CM_result':[],'CM_time': []}

sample_sizes = [50000, 100000, 500000, 1000000]
for i in sample_sizes:
    for missing_factor in [0.001, 0.01, 0.05, 0.1]:
        for num_feature in [10, 100, 500, 1000]:
            # call test() function with updated n_features and missing_factor values
            n_samples,n_features,missing_factor, n_missing,CM_result, CM_time = test(n_samples=i,n_features=num_feature, missing_factor=missing_factor)
            results_dict['n_samples'].append(n_samples)
            results_dict['missing_factor'].append(missing_factor)
            results_dict['n_features'].append(n_features)
            results_dict['n_missing'].append(n_missing)
            results_dict['CM_result'].append(CM_result)
            results_dict['CM_time'].append(CM_time)
        


# In[ ]:


result_df = pd.DataFrame(results_dict)
result_df.to_csv('kernelSVM_scan_newMF.csv', index=False)

