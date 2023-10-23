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


# In[2]:


def generate_mock_data(n_samples, n_features, n_missing):
    features, target, coef = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features//5, coef=True, random_state=42)
    ind = np.random.choice(features.shape[0], n_missing,replace = False)
    not_ind = [x for x in range(features.shape[0]) if x not in ind ]
    rand_feature = np.random.choice(np.where(coef == 0)[0], replace = False)
    mod_features = np.copy(features)
    mod_features[ind,rand_feature] = None
    #print('generator coeff', coef)
    comb = np.concatenate((mod_features, target.reshape(-1,1)), axis=1)
    comb_df = pd.DataFrame(comb, columns=[i+1 for i in range(n_features)] + ['target'])
    return comb_df,csr_matrix(features[not_ind,:]),np.ravel(target[not_ind]),csr_matrix(features[ind,:]),np.ravel(target[ind]),csr_matrix(features),ind, not_ind, features.shape[0]


# In[3]:


def get_submatrix(data):
    columns_without_nulls = data.columns[data.notnull().all()]
    C = data[columns_without_nulls]
    missing = data.drop(columns_without_nulls,axis = 1)
    return missing,C

def get_Xy(data,label):
    X = data.drop(label,axis = 1)
    y = data[label]
    return X,y

def check_orthogonal(M,l):
    flag = True
    case = ''
    for i in range(M.shape[1]):
        total = 0
        for j in range(len(l)):
            if np.isnan(M.iloc[j,i]) and not np.isclose(l[j], 0,atol=1e-03):
                flag = False
                case = 'case1: ' + str(l[j])
                break
            elif not np.isnan(M.iloc[j,i]):
                #print(f'inside case2 : M:{M.iloc[j,i]}, l:{l[j]}')
                total += M.iloc[j,i] * l[j]
        if not np.isclose(total ,0, atol = 1e-03):
            flag = False
            case = 'case2: ' + str(total)
            break
    #print(case)
    return flag

def check_certain_model(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test):
    #reg = SGDRegressor(penalty = None).fit(CX,Cy)
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_test,w_bar.T) - Cy_test)
    result = check_orthogonal(missing_test,loss)
    # print('w_bar',w_bar)
    # print('loss',loss)
    return result, reg.score(CX_test,Cy_test)


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

def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=10000):
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

    clf = SGDRegressor(penalty = None)
    #clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
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
        #print("[ActiveClean Real] Accuracy ", i ,accuracy_score(y_test, ypred,normalize = True))
        #print(f"[ActiveClean Real] Iteration: {i} Accuracy: {accuracy_score(y_test, ypred,normalize = True)}")

        if len(dirtyex) < 50:
            print("[ActiveClean Real] No More Dirty Data Detected")
            print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
            return total_cleaning, 0 if clf.score(X_clean[cleanex,:],y_clean[cleanex]) is None else clf.score(X_clean[cleanex,:],y_clean[cleanex])
    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    return total_cleaning, 0 if clf.score(X_clean[cleanex,:],y_clean[cleanex]) is None else clf.score(X_clean[cleanex,:],y_clean[cleanex]) 
            

            
            


# In[5]:


def test(n_samples, n_features, missing_factor):
    n_missing = round(n_samples*missing_factor)
    df, X_clean, y_clean, X_dirty, y_dirty, X_full, indices_dirty, indices_clean,size= generate_mock_data(n_samples = n_samples,n_features = n_features,n_missing = n_missing)
    examples = np.arange(0,size,1)
    train_indices, test_indices = train_test_split(examples, test_size=0.20)
    clean_test_indices = translate_indices(test_indices,indices_clean)
    
    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]
    
    missing_train,C_train = get_submatrix(df_train)
    CX_train,Cy_train = get_Xy(C_train,'target')
    
    missing_test,C_test = get_submatrix(df_test)
    CX_test, Cy_test = get_Xy(C_test, 'target')
    CX_test = CX_test.reset_index(drop = True)
    Cy_test = Cy_test.reset_index(drop = True)
    #print(f'CX {CX} \nCy {Cy}')
    #print('data',df)
    start_time = time.time()
    result, CM_score = check_certain_model(CX_train,Cy_train,missing_train, CX_test, Cy_test,missing_test)
    end_time = time.time()
    
    time_elapsed = end_time - start_time
    CM_time =  time_elapsed
    
    start_time = time.time()
    AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),(X_clean, y_clean),(X_clean[clean_test_indices,:], y_clean[clean_test_indices]),X_full,(train_indices,indices_dirty,indices_clean))
    AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),(X_clean, y_clean),(X_clean[clean_test_indices,:], y_clean[clean_test_indices]),X_full,(train_indices,indices_dirty,indices_clean))
    AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),(X_clean, y_clean),(X_clean[clean_test_indices,:], y_clean[clean_test_indices]),X_full,(train_indices,indices_dirty,indices_clean))
    AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),(X_clean, y_clean),(X_clean[clean_test_indices,:], y_clean[clean_test_indices]),X_full,(train_indices,indices_dirty,indices_clean))
    AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),(X_clean, y_clean),(X_clean[clean_test_indices,:], y_clean[clean_test_indices]),X_full,(train_indices,indices_dirty,indices_clean))
    end_time = time.time()
    AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
    AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    time_elapsed = end_time - start_time
    AC_time =  time_elapsed / 5    
    
    
    #print(f'N_samples: {n_samples} N_features: {n_features} N_dirty: {n_missing}')
    
    return n_samples,n_features,missing_factor,n_missing,result, CM_score, CM_time, AC_time, AC_records, AC_score


# In[6]:


results_dict = {'n_samples': [], 'missing_factor': [], 'n_features': [], 'n_missing': [],'CM_result':[],'CM_score': [],'CM_time': [], 'AC_time': [], 'AC_records_cleaned': [], 'AC_score' :[]}

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
        


# In[7]:


result_df = pd.DataFrame(results_dict)
result_df.to_csv('LR_synthetic_scan_newMF.csv', index=False) 


# In[ ]:




