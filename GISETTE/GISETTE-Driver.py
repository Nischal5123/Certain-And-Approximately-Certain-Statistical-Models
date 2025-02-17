#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import time
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer



# def get_simple_imputer_model(X_train,y_train,X_test,y_test):
#     # Assuming X_train is your training data
#     feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else None
#     clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True,)
#     clf.fit(X_train.values,y_train.values)
#     score=clf.score(X_test,y_test)
#     return score

def get_Xy(data,label):
    X = data.drop(label,axis = 1)
    y = data[label]
    return X,y

def check_certain_model(X_train, y_train, X_test, y_test):
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
    # print(X_train_complete.shape)
    # Create and train the SVM model using SGDClassifier
    svm_model =SGDClassifier(loss="hinge", alpha=1e-6, max_iter=200, fit_intercept=True, warm_start=True, random_state=42)

    # Train the model on the data without missing values
    svm_model.fit(X_train_complete, y_train_complete)

    # Extract the feature weights (model parameters)
    feature_weights = svm_model.coef_[0]

    # Check if the absolute value of feature_weights[i] is small enough for all i with missing columns
    for i in missing_columns_indices:
        if abs(feature_weights[i]) >= 1e-3:
            res = False
            #print("weight", feature_weights[i])
            break
            # Return False as soon as a condition is not met

    # Check the condition for all rows in X_train_missing_rows
    for i in range(len(X_train_missing_rows)):
        row = X_train_missing_rows[i]
        label = y_train_missing_rows[i]
        dot_product = np.sum(row[~np.isnan(row)] * feature_weights[~np.isnan(row)])
        if label * dot_product <= 1:
            #print("dot product", label * dot_product)
            res = False
            break
            # Return False if the condition is not met for any row
    if res:
        cm_score=svm_model.score(X_test,y_test)
    else:
        cm_score=0.000000000001
    # If all conditions are met, return True
    return res, cm_score


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
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True, random_state=42)
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
        ypred = clf.predict(X_test.values)
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


# Function to find a seed for a certain model
def certain_model_seed(df_train, max_attempts, noise_level,trial,X_test,y_test):
    found_seed = None
    found_score=None
    found_time=None
    total_time = 0
    total_count = 0
    average_time = 0
    total_score=0
    exact_found_time=0

    certain_model_random_seed=None

    while max_attempts > 0:
        res=False
        random_seed = np.random.randint(1, 10000)
        df_dirty = make_dirty(df_train, random_seed, noise_level)
        df_dirty.reset_index(drop=True, inplace=True)
        X_train = df_dirty.iloc[:, :-1]
        y_train = df_dirty.iloc[:, -1]

        start_time = time.time()
        found_start_time = start_time
        res, score = check_certain_model(X_train.values, y_train.values,X_test,y_test)
        end_time = time.time()
        found_end_time = end_time
        exact_found_time = found_end_time - found_start_time
        total_time += (end_time - start_time)
        total_count += 1
        print(random_seed)
        if res==True:
            found_seed = random_seed
            found_score=score
            found_time=exact_found_time
            print("Found the desired outcome with seed:", random_seed)
            break

        max_attempts -= 1

    # Write the found seed to a text file
    if found_seed is not None:
        certain_model_random_seed = found_seed
        average_time = total_time / total_count
        filename = f'GISETTE_{noise_level}_{trial}.txt'

        with open(filename, 'w') as file:
            file.write(f"Found_seed: {found_seed}\n")
            file.write(f"CM Attempts: {10000 - max_attempts}\n")
            file.write(f"No Exist Running Time(CM):  {average_time}\n")
            file.write(f"Accuracy (CM):  {found_score}\n")
            file.write(f"Exist Running Time (CM):  {found_time}\n")

    if found_seed is None and max_attempts<1:
            certain_model_random_seed = found_seed
            average_time = total_time / total_count
            filename = f'GISETTE_{noise_level}_{trial}.txt'

            with open(filename, 'w') as file:
                file.write(f"Found_seed: {found_seed}\n")
                file.write(f"CM Attempts: {10000 - max_attempts}\n")
                file.write(f"No Exist Running Time(CM):  {average_time}\n")
                file.write(f"Accuracy (CM):  {None}\n")
                file.write(f"Exist Running Time (CM):  {None}\n")

    return certain_model_random_seed

def simple_imputer_test(df_train,found_seed, noise_level,trial,X_test,y_test):
    ############ Simple Imputer Implementation #################################
    df_dirty = make_dirty(df_train, found_seed, noise_level)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]

    start_time_s = time.time()
    # Get all column names with nulls
    columns_with_nulls = X_train.columns[X_train.isnull().any()]

    # Simple imputation using mean strategy for each column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    for col in columns_with_nulls:
        X_train.loc[:, col] = imputer.fit_transform(X_train[col].values.reshape(-1, 1))[:, 0]

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()

    clf = SGDClassifier(loss="hinge", alpha=1e-6, max_iter=200, fit_intercept=True, warm_start=True,
                        random_state=42)
    clf.fit(X_train, y_train)
    simple_score = clf.score(X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    print(f'Simple imputer model, time {simple_time}, score is {simple_score}')
    filename = f'GISETTE_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"Accuracy (MI): {simple_score}\n")
        file.write(f"Running Time (MI):  {simple_time}\n")
    ################################ Simple Imputer Finish ###############################################

    return None

def naive_baseline_test(df_train,found_seed, noise_level,trial,X_test,y_test):


    df_dirty = make_dirty(df_train, found_seed, noise_level)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X = df_dirty.iloc[:, :-1]
    y = df_dirty.iloc[:, -1]
    # Convert X_train and y_train to NumPy arrays
    X_train = X.values if isinstance(X, pd.DataFrame) else X
    y_train = y.values if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y
    ############ Simple Imputer Implementation #################################


    # Find rows with missing values in X_train
    missing_rows_indices = np.where(pd.DataFrame(X_train).isnull().any(axis=1))[0]

    # Remove rows with missing values from X_train and corresponding labels from y_train
    X_train_complete = np.delete(X_train, missing_rows_indices, axis=0)
    y_train_complete = np.delete(y_train, missing_rows_indices, axis=0)

    start_time_naive = time.time()
    clf = SGDClassifier(loss="hinge", alpha=1e-6, max_iter=200, fit_intercept=True, warm_start=True, random_state=42)
    clf.fit(X_train_complete, y_train_complete)
    simple_score = clf.score(X_test, y_test)
    end_time_naive = time.time()
    ################################ Naive Imputer Finish ###############################################

    naive_time = end_time_naive - start_time_naive
    print(f'Naive Model, time {naive_time}, score is {simple_score}')
    filename = f'GISETTE_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"Accuracy (NI): {simple_score}\n")
        file.write(f"Running Time (NI):  {naive_time}\n")



def get_midas_imputer_model_classification(df_train, df_test, label):
    df_dirty = make_dirty(df_train, found_seed, noise_level)
    X_train, y_train = get_Xy(df_dirty, label)
    X_test, y_test=get_Xy(df_test,label)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='ignore')
    # after those are taken care of we can drop the columns that are still object
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    start_time_s = time.time()
    imputer = md.Midas(layer_structure=[256, 256], vae_layer=False, seed=89, input_drop=0.75)
    imputer.build_model(X_train, softmax_columns=categorical_columns)
    imputer.train_model(training_epochs=1)
    imputations = imputer.generate_samples(m=1).output_list
    X_train = imputations[0]

    clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_train, y_train)
    midas_score = clf.score(X_test, y_test)
    end_time_s = time.time()
    midas_time = end_time_s - start_time_s
    print(f'Midas Model, time {midas_time}, score is {midas_score}')
    filename = f'GISETTE_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"Midas Accuracy: {midas_score}\n")
        file.write(f"Midas running time:  {midas_time}\n")
    return midas_score, midas_time



def active_clean_driver(data,seed,noise_level,trial,df_test):
    df_dirty = make_dirty(data, seed, noise_level)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)

    features_test, target_test, X_clean, y_clean, X_dirty, y_dirty, X_full, train_indices, indices_dirty, indices_clean = genreate_AC_data(
        df_dirty, df_test)

    start_time = time.time()
    AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
                                           (X_clean, y_clean),
                                           (features_test, target_test),
                                           X_full,
                                           (train_indices, indices_dirty, indices_clean))
    AC_records_2, AC_score_2 = activeclean((X_dirty, y_dirty),
                                           (X_clean, y_clean),
                                           (features_test, target_test),
                                           X_full,
                                           (train_indices, indices_dirty, indices_clean))
    AC_records_3, AC_score_3 = activeclean((X_dirty, y_dirty),
                                           (X_clean, y_clean),
                                           (features_test, target_test),
                                           X_full,
                                           (train_indices, indices_dirty, indices_clean))
    AC_records_4, AC_score_4 = activeclean((X_dirty, y_dirty),
                                           (X_clean, y_clean),
                                           (features_test, target_test),
                                           X_full,
                                           (train_indices, indices_dirty, indices_clean))
    AC_records_5, AC_score_5 = activeclean((X_dirty, y_dirty),
                                           (X_clean, y_clean),
                                           (features_test, target_test),
                                           X_full,
                                           (train_indices, indices_dirty, indices_clean))
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    AC_time = elapsed_time / 5

    AC_records = (AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5) / 5
    AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    filename = f'GISETTE_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"Number of Examples Cleaned (AC): {AC_records}\n")
        file.write(f"Running Time (AC):  {AC_time}\n")
        file.write(f"Accuracy (AC):  {AC_score}\n")
    return None


if __name__ == '__main__':
    for noise_level in [0.001,0.01]:
        for trial in [1,2,3]:
            print("###############################       Started Missing Factor : {}..  Trial : {}  ########################################".format(noise_level,trial))
            found_seed=None
            # Load the training data and labels
            X_train_scaled = np.loadtxt('gisette_train.data')
            y_train = np.loadtxt('gisette_train.labels')

            # Load the validation data and labels
            X_test_scaled = np.loadtxt('gisette_valid.data')
            y_test = np.loadtxt('gisette_valid.labels')


            # # Instantiate the MinMaxScaler
            # scaler = MinMaxScaler()
            #
            # # Scale the training data
            # X_train_scaled = scaler.fit_transform(X_train)
            #
            # # Scale the test data
            # X_test_scaled = scaler.transform(X_test)

            # Create DataFrames for training and test
            df_train = pd.DataFrame(data=np.column_stack((X_train_scaled, y_train)), columns=[f'feature_{i}' for i in range(X_train_scaled.shape[1])] + ['label'])
            df_test = pd.DataFrame(data=np.column_stack((X_test_scaled, y_test)), columns=[f'feature_{i}' for i in range(X_test_scaled.shape[1])] + ['label'])
            max_attempts=10000
            found_seed=certain_model_seed(df_train, max_attempts, noise_level,trial, X_test_scaled, y_test)
            if found_seed != None:
                simple_imputer_test(df_train, found_seed, noise_level, trial, X_test_scaled, y_test)
                naive_baseline_test(df_train, found_seed, noise_level, trial, X_test_scaled, y_test)
                active_clean_driver(df_train, found_seed, noise_level, trial, df_test)
                #get_midas_imputer_model_classification(df_train, df_test, 'label')
            else:
                found_seed= np.random.randint(1, 10000)
                simple_imputer_test(df_train, found_seed, noise_level, trial, X_test_scaled, y_test)
                naive_baseline_test(df_train, found_seed, noise_level, trial, X_test_scaled, y_test)
                active_clean_driver(df_train, found_seed, noise_level, trial, df_test)
                #get_midas_imputer_model_classification(df_train, df_test, 'label')
            print("###############################       Complete Missing Factor : {}..  Trial : {}    #####################################".format(noise_level, trial))




