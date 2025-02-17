#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


def check_certain_model(X_train, y_train):
    res = True
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    # Convert X_train and y_train to NumPy arrays
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train

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
    #print(X_train_complete.shape)
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


# In[4]:


def make_dirty(df, random_seed, missing_factor):
    np.random.seed(random_seed) 
    num_cols = df.shape[1]
    num_dirty_cols = 1

    num_rows = df.shape[0]
    num_dirty_rows = int(missing_factor * num_rows)
    
    dirty_cols = np.random.choice(df.columns[:-1], num_dirty_cols, replace=False)

    df_dirty = df.copy()

    for col in dirty_cols:
        dirty_rows = np.random.choice(num_rows, num_dirty_rows, replace = False)
        df_dirty.loc[dirty_rows, col] = np.nan

    return df_dirty





# In[5]:


def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]

def error_classifier(total_labels, full_data):
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices,:],labels)
        return clf
    else:
        return None

def ec_filter(dirtyex, full_data, clf, t=0.90):
    if clf != None:
        pred = clf.predict_proba(full_data[dirtyex,:])
        return [j for i,j in enumerate(dirtyex) if pred[i][0] < t]

    print("CLF none")

    return dirtyex


def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=1000000):
    #makes sure the initialization uses the training data
    X = dirty_data[0][translate_indices(indextuple[0],indextuple[1]),:]
    y = dirty_data[1][translate_indices(indextuple[0],indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    #print("[ActiveClean Real] Initialization")

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    total_labels = []
    total_cleaning = 0  # Initialize the total count of missing or originally dirty examples


    ##Not in the paper but this initialization seems to work better, do a smarter initialization than
    ##just random sampling (use random initialization)
    topbatch = np.random.choice(range(0,len(dirtyex)), batchsize)
    examples_real = [dirtyex[j] for j in topbatch]
    examples_map = translate_indices(examples_real, indextuple[2])


    #Apply Cleaning to the Initial Batch
    cleanex.extend(examples_map)
    for j in set(examples_real):
        dirtyex.remove(j)

    #clf = SGDRegressor(penalty = None)
    clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_clean[cleanex,:],y_clean[cleanex])

    for i in range(50, total, batchsize):
        #print("[ActiveClean Real] Number Cleaned So Far ", len(cleanex))
        ypred = clf.predict(X_test)
        #print("[ActiveClean Real] Prediction Freqs",np.sum(ypred), np.shape(ypred))
        #print(f"[ActiveClean Real] Prediction Freqs sum ypred: {np.sum(ypred)} shape of ypred: {np.shape(ypred)}")
        #print(classification_report(y_test, ypred))

        #Sample a new batch of data
        examples_real = np.random.choice(dirtyex, batchsize)
        
        # Calculate the count of missing or originally dirty examples within the batch
        missing_count = sum(1 for r in examples_real if r in indextuple[1])
        total_cleaning += missing_count  # Add the count to the running total
        
        
        examples_map = translate_indices(examples_real, indextuple[2])

        total_labels.extend([(r, (r in lset)) for r in examples_real])

        #on prev. cleaned data train error classifier
        ec = error_classifier(total_labels, full_data)
        print(ec)
        for j in examples_real:
            try:
                dirtyex.remove(j)
            except ValueError:
                pass

        dirtyex = ec_filter(dirtyex, full_data, ec)

        #Add Clean Data to The Dataset
        cleanex.extend(examples_map)

        #uses partial fit (not in the paper--not exactly SGD)
        clf.partial_fit(X_clean[cleanex,:],y_clean[cleanex])

        print('Clean',len(cleanex))
        # print("[ActiveClean Real] Accuracy ", i ,accuracy_score(y_test, ypred,normalize = True))
        # print(f"[ActiveClean Real] Iteration: {i} Accuracy: {accuracy_score(y_test, ypred,normalize = True)}")

        if len(dirtyex) < 50:
            print("[ActiveClean Real] No More Dirty Data Detected")
            print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
            return total_cleaning, 0 if clf.score(X_clean[cleanex,:],y_clean[cleanex]) is None else clf.score(X_clean[cleanex,:],y_clean[cleanex])
    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    return total_cleaning, 0 if clf.score(X_clean[cleanex,:],y_clean[cleanex]) is None else clf.score(X_clean[cleanex,:],y_clean[cleanex]) 
            


# In[6]:


def genreate_AC_data(df_train, df_test):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    features, target = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    features_test, target_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
    ind = list(features[features.isna().any(axis=1)].index)
    not_ind = list(set(range(features.shape[0])) - set(ind))
    feat = np.where(df_train.isnull().any())[0]
    e_feat = np.copy(features)
    for i in ind:
        for j in feat:
            e_feat[i, j] = 0.01 * np.random.rand()
    return features_test, target_test,csr_matrix(e_feat[not_ind,:]),np.ravel(target[not_ind]),csr_matrix(e_feat[ind,:]),np.ravel(target[ind]),csr_matrix(e_feat), np.arange(len(e_feat)).tolist(),ind, not_ind






# In[7]:


# Load the training data and labels
X_train = np.loadtxt('gisette_train.data')
y_train = np.loadtxt('gisette_train.labels')

# Load the validation data and labels
X_test = np.loadtxt('gisette_valid.data')
y_test = np.loadtxt('gisette_valid.labels')


# Instantiate the MinMaxScaler
scaler = MinMaxScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Create DataFrames for training and test
df_train = pd.DataFrame(data=np.column_stack((X_train_scaled, y_train)), columns=[f'feature_{i}' for i in range(X_train_scaled.shape[1])] + ['label'])
df_test = pd.DataFrame(data=np.column_stack((X_test_scaled, y_test)), columns=[f'feature_{i}' for i in range(X_test_scaled.shape[1])] + ['label'])



# In[7]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.001)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial1_0001.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.001)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial1_0001.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[13]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.01)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial1_001.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")


df_dirty = make_dirty(df_train, found_seed_1, 0.01)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial1_001.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")
# In[8]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.001)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial2_0001.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")


df_dirty = make_dirty(df_train, found_seed_1, 0.001)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial2_0001.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[11]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.001)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial3_0001.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.001)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial3_0001.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")
# In[ ]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.01)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial2_001.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.01)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial2_001.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[ ]:
max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.01)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial3_001.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.01)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial3_001.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.05)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial1_005.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")


df_dirty = make_dirty(df_train, found_seed_1, 0.05)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial1_005.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[ ]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.05)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial2_005.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.05)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial2_005.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[ ]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.05)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial3_005.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.05)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial3_005.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[ ]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.1)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial1_01.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.1)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial1_01.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")
# In[ ]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.1)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial2_01.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.1)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial2_01.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")

# In[ ]:


max_attempts = 10000  # Define a maximum number of attempts to avoid infinite loops   
found_seed_1 = None  # Initialize found_seed to None
total_time = 0
total_count = 0
average_time = 0
certain_model_random_seed = 0
while max_attempts > 0:
    random_seed = np.random.randint(1, 10000)  # Randomly sample a seed
    df_dirty = make_dirty(df_train, random_seed, 0.1)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]
    
    try:
        start_time = time.time()
        res, feature_weights = check_certain_model(X_train.values, y_train.values)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_count += 1
    except Exception as e:
        print(f"Error in check_certain_model: {e}")
        max_attempts -= 1
        continue  # Continue to the next iteration
    
    print(random_seed)
    if res is True:
        found_seed_1 = random_seed  # Store the found seed
        print("Found the desired outcome with seed:", random_seed)
        break
    
    max_attempts -= 1

# Write the found seed to a text file
if found_seed_1 is not None:
    certain_model_random_seed = found_seed_1
    average_time = total_time / total_count
    with open('GISETTE_certain_model_seed_trial3_01.txt', 'w') as file:
        file.write(f"found_seed_1: {found_seed_1}\n")
        file.write(f"attempts: {10000 - max_attempts}\n")
        file.write(f"time spent from certain model algorithm when NO certain model eixts:  {average_time}\n")

df_dirty = make_dirty(df_train, found_seed_1, 0.1)
# Access the last column by its numerical index (-1)
df_dirty.reset_index(drop=True, inplace=True)

features_test, target_test,X_clean,y_clean,X_dirty,y_dirty,X_full,train_indices,indices_dirty,indices_clean=genreate_AC_data(df_dirty, df_test)

start_time = time.time()
AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (features_test, target_test),
            X_full,
            (train_indices,indices_dirty,indices_clean))
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
AC_time =  elapsed_time / 5 

AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    
with open('AC_trial3_01.txt', 'w') as file:
    file.write(f"AC example cleaned: {AC_records}\n")
    file.write(f"AC running time:  {AC_time}\n")