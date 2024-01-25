#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer
from utils import drop_categorical_columns, drop_label_with_null

# In[2]:

def get_simple_imputer_model(X_train,y_train,X_test,y_test):
    clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)
    return score

def get_Xy(data,label):
    X = data.drop(label,axis = 1)
    y = data[label]
    return X,y

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


# In[3]:


def make_dirty(df, random_seed, missing_factor):
    np.random.seed(random_seed)
    num_cols = df.shape[1]
    num_dirty_cols = 1

    num_rows = df.shape[0]
    num_dirty_rows = int(missing_factor * num_rows)

    dirty_cols = np.random.choice(df.columns[:-1], num_dirty_cols, replace=False)

    df_dirty = df.copy()

    for col in dirty_cols:
        dirty_rows = np.random.choice(num_rows, num_dirty_rows, replace=False)
        df_dirty.loc[dirty_rows, col] = np.nan

    return df_dirty


# In[ ]:


def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s, t in enumerate(imap) if t in lset]


def error_classifier(total_labels, full_data):
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices, :], labels)
        return clf
    else:
        return None


def ec_filter(dirtyex, full_data, clf, t=0.90):
    if clf != None:
        pred = clf.predict_proba(full_data[dirtyex, :])
        return [j for i, j in enumerate(dirtyex) if pred[i][0] < t]

    print("CLF none")

    return dirtyex


def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=1000000):
    # makes sure the initialization uses the training data
    X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
    y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    # print("[ActiveClean Real] Initialization")

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    total_labels = []
    total_cleaning = 0  # Initialize the total count of missing or originally dirty examples

    ##Not in the paper but this initialization seems to work better, do a smarter initialization than
    ##just random sampling (use random initialization)
    topbatch = np.random.choice(range(0, len(dirtyex)), batchsize)
    examples_real = [dirtyex[j] for j in topbatch]
    examples_map = translate_indices(examples_real, indextuple[2])

    # Apply Cleaning to the Initial Batch
    cleanex.extend(examples_map)
    for j in set(examples_real):
        dirtyex.remove(j)

    # clf = SGDRegressor(penalty = None)
    clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_clean[cleanex, :], y_clean[cleanex])

    for i in range(50, total, batchsize):
        # print("[ActiveClean Real] Number Cleaned So Far ", len(cleanex))
        ypred = clf.predict(X_test)
        # print("[ActiveClean Real] Prediction Freqs",np.sum(ypred), np.shape(ypred))
        # print(f"[ActiveClean Real] Prediction Freqs sum ypred: {np.sum(ypred)} shape of ypred: {np.shape(ypred)}")
        # print(classification_report(y_test, ypred))

        # Sample a new batch of data
        examples_real = np.random.choice(dirtyex, batchsize)

        # Calculate the count of missing or originally dirty examples within the batch
        missing_count = sum(1 for r in examples_real if r in indextuple[1])
        total_cleaning += missing_count  # Add the count to the running total

        examples_map = translate_indices(examples_real, indextuple[2])

        total_labels.extend([(r, (r in lset)) for r in examples_real])

        # on prev. cleaned data train error classifier
        ec = error_classifier(total_labels, full_data)
        print(ec)
        for j in examples_real:
            try:
                dirtyex.remove(j)
            except ValueError:
                pass

        dirtyex = ec_filter(dirtyex, full_data, ec)

        # Add Clean Data to The Dataset
        cleanex.extend(examples_map)

        # uses partial fit (not in the paper--not exactly SGD)
        clf.partial_fit(X_clean[cleanex, :], y_clean[cleanex])

        print('Clean', len(cleanex))
        # print("[ActiveClean Real] Accuracy ", i ,accuracy_score(y_test, ypred,normalize = True))
        # print(f"[ActiveClean Real] Iteration: {i} Accuracy: {accuracy_score(y_test, ypred,normalize = True)}")

        if len(dirtyex) < 50:
            print("[ActiveClean Real] No More Dirty Data Detected")
            print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
            return total_cleaning, 0 if clf.score(X_clean[cleanex, :], y_clean[cleanex]) is None else clf.score(
                X_clean[cleanex, :], y_clean[cleanex])
    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    return total_cleaning, 0 if clf.score(X_clean[cleanex, :], y_clean[cleanex]) is None else clf.score(
        X_clean[cleanex, :], y_clean[cleanex])


# In[ ]:


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
    return features_test, target_test, csr_matrix(e_feat[not_ind, :]), np.ravel(target[not_ind]), csr_matrix(
        e_feat[ind, :]), np.ravel(target[ind]), csr_matrix(e_feat), np.arange(len(e_feat)).tolist(), ind, not_ind

def certain_model_seed(df_train, max_attempts, noise_level,trial):
    found_seed = None
    total_time = 0
    total_count = 0
    average_time = 0
    certain_model_random_seed=None

    while max_attempts > 0:
        found_start_time = 0
        found_end_time = 0
        random_seed = np.random.randint(1, 10000)
        df_dirty = make_dirty(df_train, random_seed, noise_level)
        df_dirty.reset_index(drop=True, inplace=True)
        X_train = df_dirty.iloc[:, :-1]
        y_train = df_dirty.iloc[:, -1]


        try:
            start_time = time.time()
            found_start_time = start_time
            res, _ = check_certain_model(X_train.values, y_train.values)
            end_time = time.time()
            found_end_time = end_time
            total_time += (end_time - start_time)
            total_count += 1
        except Exception as e:
            print(f"Error in check_certain_model: {e}")
            max_attempts -= 1
            continue

        print(random_seed)
        if res:
            found_seed = random_seed
            print("Found the desired outcome with seed:", random_seed)
            break

        max_attempts -= 1

    # Write the found seed to a text file
    if found_seed is not None:
        certain_model_random_seed = found_seed
        average_time = total_time / total_count
        filename = f'TUANDROMD_{noise_level}_{trial}.txt'

        with open(filename, 'w') as file:
            file.write(f"Found_seed: {found_seed}\n")
            file.write(f"CM Attempts: {10000 - max_attempts}\n")
            file.write(f"CM no exist:  {average_time}\n")
            file.write(f"CM exist:  {found_end_time - found_start_time}\n")

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

    simple_score = get_simple_imputer_model(X_train, y_train, X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    print(f'Simple imputer model, time {simple_time}, score is {simple_score}')
    filename = f'TUANDROMD_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"MI Accuracy: {simple_score}\n")
        file.write(f"MI running time:  {simple_time}\n")
    ################################ Simple Imputer Finish ###############################################

    return None

def naive_baseline_test(df_train,found_seed, noise_level,trial,X_test,y_test):
    ############ Simple Imputer Implementation #################################
    df_dirty = make_dirty(df_train, found_seed, noise_level)
    # Access the last column by its numerical index (-1)
    df_dirty.reset_index(drop=True, inplace=True)
    X_train = df_dirty.iloc[:, :-1]
    y_train = df_dirty.iloc[:, -1]

    start_time_s = time.time()
    # Create a copy of X_train
    X_train = X_train.copy()

    # Drop rows with null values in the copy of X_train
    X_train.dropna(inplace=True)

    # Now X_train is a copy and not a view, and modifications won't raise the warning

    # Align y_train with the modified X_train
    y_train = y_train.iloc[X_train.index]

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()

    simple_score = get_simple_imputer_model(X_train, y_train, X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    print(f'Naive Model, time {simple_time}, score is {simple_score}')
    filename = f'TUANDROMD_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"NM Accuracy: {simple_score}\n")
        file.write(f"NM running time:  {simple_time}\n")
    ################################ Simple Imputer Finish ###############################################

    return None


def get_midas_imputer_model_classification(df_train, df_test, label):
    df_dirty = make_dirty(df_train, found_seed, noise_level)
    X_train, y_train = get_Xy(df_dirty, label)
    X_test, y_test = get_Xy(df_test, label)
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
    filename = f'T{noise_level}_{trial}.txt'
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
    filename = f'TUANDROMD_{noise_level}_{trial}.txt'
    with open(filename, 'a+') as file:
        file.write(f"AC example cleaned: {AC_records}\n")
        file.write(f"AC running time:  {AC_time}\n")
        file.write(f"AC score :  {AC_score}\n")
    return None


if __name__ == '__main__':

            # Replace 'path_to_csv' with the actual path to your CSV file
            file_path = 'Final-Datasets/COVID.csv'

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Get the index of the last column (assuming it's the last column in the DataFrame)
            last_column_index = df.columns[-1]

            # Replace 'malware' with 1 and 'goodware' with -1 in the last column
            df[last_column_index] = df[last_column_index].replace({'malware': 1, 'goodware': -1})
            columns = [f'feature_{i}' for i in range(df.shape[1]-1)] + ['label']
            df.columns=columns

            # We saw a row (#2533) with all the values missing in the row. We believe this row was written in by accident, and is not part of the data becucase the dataset claims that it does NOT has missing values (https://doi.org/10.24432/C5560H)
            # As a result, we delete this row
            df = df.dropna(how='all')

            # Split the DataFrame into features (X) and labels (y)
            X = df.iloc[:, :-1]  # Features (all columns except the last one)
            y = df.iloc[:, -1]  # Labels (the last column)

            # Split the data into training and testing sets (adjust the test_size as needed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Merge X_train and y_train into df_train
            df_train = pd.concat([X_train, y_train], axis=1)

            # Merge X_test and y_test into df_test
            df_test = pd.concat([X_test, y_test], axis=1)

            # Reset the index for df_train and df_test
            df_train.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)

            max_attempts = 10000
            found_seed = certain_model_seed(df_train, max_attempts)

            simple_imputer_test(df_train, found_seed, X_test, y_test)
            naive_baseline_test(df_train, found_seed, X_test, y_test)
            active_clean_driver(df_train, found_seed,  df_test)
            get_midas_imputer_model_classification(df_train, df_test, 'label')
