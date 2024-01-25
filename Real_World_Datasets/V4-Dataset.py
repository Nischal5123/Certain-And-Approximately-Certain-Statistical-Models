import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from utils import missing_values_table
from utils import drop_categorical_columns
from utils import get_single_value_columns
from utils import drop_label_with_null
from ActiveClean import activeclean
from utils import split_features_labels
import time
import os
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
def get_classification_complete_test_data(X_val,Y_val,model):
    # Convert X_train and y_train to NumPy arrays
    X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    Y_val = Y_val.values if isinstance(Y_val, pd.DataFrame) or isinstance(Y_val, pd.Series) else Y_val

    # Find indices of columns with missing values in X_train
    missing_columns_indices = np.where(pd.DataFrame(X_val).isnull().any(axis=0))[0]

    # Find rows with missing values in X_train
    missing_rows_indices = np.where(pd.DataFrame(X_val).isnull().any(axis=1))[0]

    # Record the rows with missing values and their corresponding y_train values
    X_val_missing_rows = X_val[missing_rows_indices]
    y_val_missing_rows = Y_val[missing_rows_indices]


    # Remove rows with missing values from X_train and corresponding labels from y_train
    X_val_complete = np.delete(X_val, missing_rows_indices, axis=0)
    y_val_complete = np.delete(Y_val, missing_rows_indices, axis=0)

    return accuracy_score(y_val_complete,model.predict(X_val_complete))

def check_certain_model_classification(X_t, y_t, X_val, Y_val):
    X_train=X_t
    y_train=y_t
    res = True
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    test_accuracy=0

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


    print(f"Original shape: X_train: {X_train.shape}, X_train: {X_train_complete.shape}")
    # print(f"Shape after removal: X_train: {X_train_complete.shape}, y_train: {y_train_complete.shape}")



    # Create and train the SVM model using SGDClassifier
    svm_model = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)

    # Train the model on the data without missing values
    svm_model.fit(X_train_complete, y_train_complete)
    test_accuracy=get_classification_complete_test_data(X_val,Y_val,svm_model)

    # Extract the feature weights (model parameters)
    feature_weights = svm_model.coef_[0]
    #print(feature_weights)

    # Check if the absolute value of feature_weights[i] is small enough for all i with missing columns
    for i in missing_columns_indices:
        if abs(feature_weights[i]) >= 1e-03:
            res = False
            print('Case 1')
            break
            # Return False as soon as a condition is not met

    # Check the condition for all rows in X_train_missing_rows
    for i in range(len(X_train_missing_rows)):
        row = X_train_missing_rows[i]
        label = y_train_missing_rows[i]
        dot_product = np.sum(row[~np.isnan(row)] * feature_weights[~np.isnan(row)])
        if label * dot_product <= 1:
            res = False
            print('Case 1')
            break
            # Return False if the condition is not met for any row
    return res, test_accuracy

def get_submatrix(data):
    columns_without_nulls = data.columns[data.notnull().all()]
    C = data[columns_without_nulls]
    missing = data.drop(columns_without_nulls,axis = 1)
    return missing,C

def get_Xy(data,label):
    X = data.drop(label,axis = 1)
    y = data[label]
    return X,y


def genreate_AC_data(df_train, df_test,label):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    features, target = get_Xy(df_train, label)
    features_test, target_test = get_Xy(df_test, label)
    ind = list(features[features.isna().any(axis=1)].index)
    not_ind = list(set(range(features.shape[0])) - set(ind))
    feat = np.where(df_train.isnull().any())[0]
    e_feat = np.copy(features)
    for i in ind:
        for j in feat:
            e_feat[i, j] = 0.01 * np.random.rand()
    return features_test, target_test,csr_matrix(e_feat[not_ind,:]),np.ravel(target[not_ind]),csr_matrix(e_feat[ind,:]),np.ravel(target[ind]),csr_matrix(e_feat), np.arange(len(e_feat)).tolist(),ind, not_ind
def finding_certain_model_classification(dataset):
    data = dataset
    label='84'
    df_dropped = drop_categorical_columns(dataset.copy())
    df = drop_label_with_null(df_dropped, label)
    del df_dropped
    print(missing_values_table(df))
    #print(f'##########################     Number of rows with missing values {df.isnull().any(axis=1).sum()} ##########################')
    #classification for label 3 or not ie Manufacturing or not
    X, y = split_features_labels(df, label, 'classification')

    # Split the data into training and testing sets (adjust the test_size as needed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merge X_train and y_train into df_train
    df_train = pd.concat([X_train, Y_train], axis=1)

    # Merge X_test and y_test into df_test
    df_test = pd.concat([X_test, Y_test], axis=1)
    df_test.dropna(inplace=True)

    # Reset the index for df_train and df_test
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    missing_train, C_train = get_submatrix(df_train)
    CX_train, Cy_train = get_Xy(C_train, label)

    missing_test, C_test = get_submatrix(df_test)
    CX_test, Cy_test = get_Xy(C_test, label)
    CX_test = CX_test.reset_index(drop=True)
    Cy_test = Cy_test.reset_index(drop=True)

    start_time = time.time()
    # Instantiate the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the training data
    X_train = scaler.fit_transform(X_train)

    # Scale the test data
    X_test = scaler.transform(X_test)
    try:
        result, CM_score = check_certain_model_classification(X_train,Y_train,X_test,Y_test)
    except ValueError as e:
        print(e)
        result=False
    end_time = time.time()
    del df
    print(f'########################## Certain Model result for {label}, time :{end_time - start_time} , test accuracy {CM_score} and RESULT ###### {result} ########################## ')
    features_test, target_test, X_clean, y_clean, X_dirty, y_dirty, X_full, train_indices, indices_dirty, indices_clean = genreate_AC_data(
        df_train, df_test,'84')

    start_time = time.time()
    AC_records, AC_score = activeclean((X_dirty, y_dirty),
                                           (X_clean, y_clean),
                                           (features_test, target_test),
                                           X_full,
                                           (train_indices, indices_dirty, indices_clean),'classification')
    end_time = time.time()

    # Calculate the elapsed time
    ac_elapsed_time = end_time - start_time
    print(
        f'########################## Records_Cleaned {AC_records}, time :{ac_elapsed_time} , test accuracy {AC_score} ')


if __name__ == '__main__':
    data_path = 'Final-Datasets/'
    csv_filename = 'combined_V4_data.csv'
    path = os.path.join(data_path, csv_filename)
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)
    if df.isnull().values.any():
            print("Converting '{}'...".format(csv_filename))
            finding_certain_model_classification(df)
