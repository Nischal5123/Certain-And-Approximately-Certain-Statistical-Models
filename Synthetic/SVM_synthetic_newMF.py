#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time
import unicodedata
import pandas as pd
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


# In[2]:


def generate_mock_data(n_samples, n_missing, n_features):
    #print(coef)
    
    random_cols = 100*np.random.uniform(low = 25, high = 100, size = (n_samples//2, n_features))
    ones_col = np.ones((n_samples//2, 1))
    original_array = np.hstack((random_cols, ones_col))
    new_array = np.copy(original_array)
    new_array[:, 1:] *= -1
    merged_array = np.concatenate((original_array, new_array), axis=0)
    features, target = merged_array[:, :-1], merged_array[:, -1]
    
    
    ind = np.random.choice(features.shape[0], n_missing,replace = False)
    not_ind = [x for x in range(features.shape[0]) if x not in ind ]
    feat = 0
    mod_feat = np.copy(features)
    mod_feat[ind,feat] = np.nan
    
    e_feat = np.copy(features)
    e_feat[ind,feat] = 0.01 * np.random.rand()
    
    # D = mod_feat[~np.isnan(mod_feat).any(axis=1)] # submatrix of X with no missing values
    # Yd = target[~np.isnan(mod_feat).any(axis=1)] # labels truncated correspondingly
    # J = np.where(np.isnan(mod_feat).any(axis=1))[0] # list of indices for training examples that have missing values
    # j = len(J) # the number of indices in list J
    # K = np.unique(np.where(np.isnan(mod_feat))[1].reshape(-1,1))
    # #K = [1]
    # k = len(K) # the number of indices in list K
    
    return mod_feat ,e_feat, target, csr_matrix(e_feat[not_ind,:]),np.ravel(target[not_ind]),csr_matrix(e_feat[ind,:]),np.ravel(target[ind]),csr_matrix(e_feat),ind, not_ind, e_feat.shape[0]


# In[3]:


#D <- X non-missing
#Yd <- y non-missing
#J <- list of indices of training examples having missing values
#K <- list of indices for feature vectors having missing values
#e_feat <- X with repairs

def check_certain_model(mod_feat, e_feat,target, train_indices, test_indices):
    
    mod_feat_train = mod_feat[train_indices]
    mod_feat_test = mod_feat[test_indices]
    
    e_feat_test = mod_feat[test_indices]
    target_test = target[test_indices]
    
    D_train = mod_feat_train[~np.isnan(mod_feat_train).any(axis=1)]
    D_test = mod_feat_test[~np.isnan(mod_feat_test).any(axis=1)]
    Yd_train = target[train_indices][~np.isnan(mod_feat_train).any(axis=1)]
    Yd_test = target[test_indices][~np.isnan(mod_feat_test).any(axis=1)]
    J_test = np.where(np.isnan(mod_feat_test).any(axis=1))[0]
    K_test = np.unique(np.where(np.isnan(mod_feat_test))[1].reshape(-1,1))

    res = True
    clf = SVC(kernel='linear')
    clf.fit(D_train, Yd_train)
    w_c = clf.coef_.flatten()
    
    
    #TEST DATA
    #K_test, J_test, e_feat_test, D_test, Yd_test
    for i in range(len(K_test)):
        if not np.isclose(w_c[K_test[i]], 0,atol=1e-03):
            print('test1 w')
            res = False
            break
            
    for i in range(len(J_test)):
        if target_test[J_test[i]]*round(max(min(np.dot(w_c, e_feat_test[J_test[i], :]), 1.0), -1.0), 1) < 1:
            print('test2',target_test[J[i]],round(max(min(np.dot(w_c, e_feat_test[J_test[i], :]), 1.0), -1.0), 1))
            res = False
            break
    #print(f'Certain Model: {res}, w_bar {w_c}')
    return res, clf.score(D_test,Yd_test)


# In[4]:


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
            


# In[5]:


def test(n_samples, n_features, missing_factor):
    n_missing = round(n_samples*missing_factor)
    
    mod_feat,e_feat,target, X_clean, y_clean, X_dirty, y_dirty, X_full, indices_dirty, indices_clean,size= generate_mock_data(n_samples = n_samples,n_missing = n_missing, n_features = n_features)
    examples = np.arange(0,size,1)
    train_indices, test_indices = train_test_split(examples, test_size=0.2, random_state = 42)
    clean_test_indices = translate_indices(test_indices,indices_clean)
    
    start_time = time.time()
    result, CM_score = check_certain_model(mod_feat, e_feat,target, train_indices, test_indices)
    end_time = time.time()
    
    time_elapsed = end_time - start_time
    CM_time =  time_elapsed
    
    start_time = time.time()
    AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
             (X_clean, y_clean),
             (X_clean[clean_test_indices,:], y_clean[clean_test_indices]),
             X_full,
             (train_indices,indices_dirty,indices_clean))
    AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
             (X_clean, y_clean),
             (X_clean[clean_test_indices,:], y_clean[clean_test_indices]),
             X_full,
             (train_indices,indices_dirty,indices_clean))
    end_time = time.time()
    AC_records = (AC_records_1 + AC_records_2) / 2
    AC_score = (AC_score_1 + AC_score_2) / 2
    time_elapsed = end_time - start_time
    AC_time =  time_elapsed / 2   


    return n_samples, n_features, missing_factor,n_missing, result, CM_score, CM_time, AC_time, AC_records, AC_score


# In[6]:


results_dict = {'n_samples': [], 'missing_factor': [], 'n_features': [], 'n_missing': [],'CM_result':[],'CM_score': [],'CM_time': [], 'AC_time': [], 'AC_records_cleaned': [], 'AC_score' :[]}

# In[7]:

sample_sizes = [100000, 500000, 1000000]
for i in sample_sizes:
    for missing_factor in [0.001, 0.01, 0.05, 0.1]:
        for num_feature in [10, 100, 500, 1000]:
        # call test() function with updated n_samples and missing_factor values
            n_samples,n_features,missing_factor, n_missing,CM_result, CM_score, CM_time, AC_time, AC_records, AC_score = test(n_samples=i,n_features=num_feature, missing_factor=missing_factor)
            results_dict['n_samples'].append(n_samples)
            results_dict['missing_factor'].append(missing_factor)
            results_dict['n_features'].append(n_features)
            results_dict['n_missing'].append(n_missing)
            results_dict['CM_result'].append(CM_result)
            results_dict['CM_time'].append(CM_time)
            results_dict['AC_time'].append(AC_time)
            results_dict['AC_records_cleaned'].append(AC_records)





# In[ ]:



result_df = pd.DataFrame(results_dict)
result_df.to_csv('SVM_example_scan_1M_newMF.csv', index=False)


# In[ ]:




