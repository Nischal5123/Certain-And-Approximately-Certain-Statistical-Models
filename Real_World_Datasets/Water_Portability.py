# In[1]:


import time
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from utils import (
    get_simple_imputer_model_classification,
    get_naive_imputer_model_classification,
    get_knn_imputer_model_classification,
    get_miwei_imputer_model_classification
)
from ActiveClean import activeclean
import os
# In[2]:


def check_certain_model(X_train, y_train, X_test, y_test):
    res = True

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
    svm_model = SGDClassifier(
        loss="hinge",
        alpha=0.0000000001,
        max_iter=10000,
        fit_intercept=True,
        warm_start=True,
        random_state=42,
    )

    # Train the model on the data without missing values
    svm_model.fit(X_train_complete, y_train_complete)

    # Extract the feature weights (model parameters)
    feature_weights = svm_model.coef_[0]

    # Check if the absolute value of feature_weights[i] is small enough for all i with missing columns
    for i in missing_columns_indices:
        if abs(feature_weights[i]) >= 1e-3:
            res = False
            # print("weight", feature_weights[i])
            break
            # Return False as soon as a condition is not met

    # Check the condition for all rows in X_train_missing_rows
    for i in range(len(X_train_missing_rows)):
        row = X_train_missing_rows[i]
        label = y_train_missing_rows[i]
        dot_product = np.sum(row[~np.isnan(row)] * feature_weights[~np.isnan(row)])
        if label * dot_product <= 1:
            # print("dot product", label * dot_product)
            res = False
            break
            # Return False if the condition is not met for any row
    if res:
        cm_score = svm_model.score(X_test, y_test)
    else:
        cm_score=0.000000000001
    # If all conditions are met, return True
    return res, cm_score


def genreate_AC_data(df_train, df_test):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    #get xy doesn't matter last column is the label
    features, target = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    features_test, target_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
    ind = list(features[features.isna().any(axis=1)].index)
    not_ind = list(set(range(features.shape[0])) - set(ind))
    feat = np.where(df_train.isnull().any())[0]
    e_feat = np.copy(features)
    for i in ind:
        for j in feat:
            e_feat[i, j] = 0.01 * np.random.rand()
    return (
        features_test,
        target_test,
        csr_matrix(e_feat[not_ind, :]),
        np.ravel(target[not_ind]),
        csr_matrix(e_feat[ind, :]),
        np.ravel(target[ind]),
        csr_matrix(e_feat),
        np.arange(len(e_feat)).tolist(),
        ind,
        not_ind,
    )


def get_Xy(data, label):
    X = data.drop(label, axis=1)
    y = data[label]
    return X, y


def active_clean_driver(df_train, df_test):
    (
        features_test,
        target_test,
        X_clean,
        y_clean,
        X_dirty,
        y_dirty,
        X_full,
        train_indices,
        indices_dirty,
        indices_clean,
    ) = genreate_AC_data(df_train, df_test)

    start_time = time.time()
    AC_records_1, AC_score_1 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_2, AC_score_2 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_3, AC_score_3 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_4, AC_score_4 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_5, AC_score_5 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    AC_time = elapsed_time / 5

    AC_records = (
        AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5
    ) / 5
    AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5
    return AC_records ,AC_score, AC_time




if __name__ == "__main__":
    df = pd.read_csv("Final-Datasets/water_potability.csv")
    label = 'Potability'
    X, y = get_Xy(
        df, 'Potability'
    )  # Features (all columns except the last one)  # Labels (the last column)

    # print(missing_values_table(df))

    # Split the data into training and testing sets (adjust the test_size as needed)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Merge X_train and y_train into df_train
    df_train = pd.concat([X_train, Y_train], axis=1)

    # Merge X_test and y_test into df_test
    df_test = pd.concat([X_test, Y_test], axis=1)

    # Reset the index for df_train and df_test
    df_train.reset_index(drop=True, inplace=True)
    # Drop all rows with null values in the original DataFrame
    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]


    # Calculate the total number of examples
    total_examples = len(X_train)
    # Count the number of missing values in each row
    missing_values_per_row = X_train.isnull().sum(axis=1)

    # Count the total number of rows with missing values
    rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])

    # Display the result
    print("Number of rows with missing values:", rows_with_missing_values)
    # Calculate the missing factor for each column
    missing_factor = rows_with_missing_values / total_examples
    print(f"Total example {X_train.shape}, MISSING FACTOR : {missing_factor}")

    start_time = time.time()
    result, CM_score = check_certain_model(X_train.values, Y_train.values,X_test.values, y_test.values)
    end_time = time.time()
    CM_time = end_time - start_time
    print(
        f"Certain Model result for {label}, time :{CM_time}  RESULT ###### {result} and score {CM_score}"
    )

    name = f"Water_CM_Exist_{result}.txt"
    filename = os.path.join('Final-Results', name)
    with open(filename, "w+") as file:
        file.write(f"Number of Rows with missing values:{rows_with_missing_values}\n")
        file.write(f"Missing Factor:{missing_factor}\n")
        file.write(f"Running Time (CM):  {CM_time}\n")
        file.write(f"Accuracy (CM):  {CM_score}\n")

    meiwei_imputer_score, meiwei_imputer_time = get_miwei_imputer_model_classification(df_train, df_test, label)
    with open(filename, "a+") as file:
        file.write(f"Accuracy (Meiewi): {meiwei_imputer_score}\n")
        file.write(f"Running Time (Meiewi):  {meiwei_imputer_time}\n")

    simple_imputer_score, simpler_imputer_time = get_simple_imputer_model_classification(
        df_train, df_test, label
    )
    print(
        f"########################## Simple Model result for {label}, time :{simpler_imputer_time} , test accuracy {simple_imputer_score}########################## "
    )

    knn_imputer_score, knn_imputer_time = get_knn_imputer_model_classification(
        df_train, df_test, label
    )
    print(
        f"########################## Simple Model result for {label}, time :{knn_imputer_score} , test accuracy {knn_imputer_time}########################## "
    )

    naive_imputer_score, naive_imputer_time = get_naive_imputer_model_classification(
        df_train, df_test, label
    )

    print(
        f"########################## Naive Model result for {label}, time :{naive_imputer_time} , test accuracy {naive_imputer_score}########################## "
    )

    AC_records, AC_score, AC_time = active_clean_driver(df_train, df_test, label)

    with open(filename, "a+") as file:
        file.write(f"Accuracy (KNN): {knn_imputer_score}\n")
        file.write(f"Running Time (KNN):  {knn_imputer_time}\n")
        file.write(f"Accuracy (MI): {simple_imputer_score}\n")
        file.write(f"Running Time (MI):  {simpler_imputer_time}\n")
        file.write(f"Accuracy (NI): {naive_imputer_score}\n")
        file.write(f"Running Time (NI):  {naive_imputer_time}\n")
        file.write(f"Number of Examples Cleaned (AC): {AC_records}\n")
        file.write(f"Running Time (AC):  {AC_time}\n")
        file.write(f"Accuracy (AC):  {AC_score}\n")

