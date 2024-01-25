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
from io import StringIO
from scipy.io import arff
import openml as oml

def check_certain_model_classification(X_t, y_t):
    X_train=X_t
    y_train=y_t
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

    print(f"Original shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape after removal: X_train: {X_train_complete.shape}, y_train: {y_train_complete.shape}")

    if X_train.shape!=X_train_complete.shape:

        # Create and train the SVM model using SGDClassifier
        svm_model = SGDClassifier(loss='hinge', max_iter=100000000, random_state=42)

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
    else:
        res=False
        feature_weights=-100
    return res, feature_weights
def check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test):
    len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns)
    #reg = SGDRegressor(penalty = None).fit(CX,Cy)
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_train,w_bar.T) - Cy_train)
    result = check_orthogonal(missing_test,loss)
    # print('w_bar',w_bar)
    # print('loss',loss)
    return result, reg.score(CX_test,Cy_test)

# def get_submatrix(data):
#     columns_without_nulls = data.columns[data.notnull().all()]
#     C = data[columns_without_nulls]
#     missing = data.drop(columns_without_nulls,axis = 1)
#     return missing,C

def get_submatrix(data):
    rows_without_nulls = data.dropna()
    C = data.loc[rows_without_nulls.index]
    missing = data.drop(index=rows_without_nulls.index)
    return missing ,C

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

def generate_mock_data(df_train, df_test, label):
    # Combine train and test DataFrames
    df = pd.concat([df_train, df_test], ignore_index=True)
    # Reset index for the combined DataFrame
    df = df.reset_index(drop=True)
    # Split features and labels for the combined data
    features, target = split_features_labels(df, label)
    # Identify columns with null values
    dirty_columns = features.columns[features.isnull().any()].tolist()
    # Create a DataFrame with clean features (no null values)
    X_clean = features.drop(dirty_columns, axis=1)
    y_clean = target
    # Create a DataFrame with dirty features (containing null values)
    X_dirty = features.copy()
    y_dirty = target
    # Convert DataFrames to sparse matrices
    X_clean_sparse = csr_matrix(X_clean.values)
    X_dirty_sparse = csr_matrix(X_dirty.values)
    X_full_sparse = csr_matrix(features.values)
    # Indices of dirty and clean data
    indices_dirty = X_dirty.index.difference(X_clean.index)
    indices_clean = X_clean.index
    # Indices of training and test data in the original df_train and df_test
    indices_train = df_train.index
    indices_test = df_test.index
    # Size of the DataFrame
    sizeofdataframe = len(X_dirty)
    return (
        df,
        X_clean_sparse,
        y_clean,
        X_dirty_sparse,
        y_dirty,
        X_full_sparse,
        indices_dirty,
        indices_clean,
        sizeofdataframe,
        indices_train,
        indices_test
    )
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
    print(missing_values_table(data))
    print(get_single_value_columns(data))

    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    # print(missing_values_table(TEST_DF))
    for i in TEST_DF.columns:
        df_dropped = drop_categorical_columns(data)
        if i not in get_single_value_columns(df_dropped):
            label = i
            df = drop_label_with_null(df_dropped, label)
            print(f'##########################     Number of rows with missing values {df.isnull().any(axis=1).sum()} ##########################')
            X, y = split_features_labels(df, label, 'classification')

            # Split the data into training and testing sets (adjust the test_size as needed)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Merge X_train and y_train into df_train
            df_train = pd.concat([X_train, Y_train], axis=1)

            # Merge X_test and y_test into df_test
            df_test = pd.concat([X_test, Y_test], axis=1)

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
            try:

                result, CM_score = check_certain_model_classification(X_train,Y_train)

            except ValueError as e:
                print(e)
                result=False
            end_time = time.time()
            print(f'########################## Certain Model result for {label}, time :{end_time - start_time} and RESULT ###### {result} ########################## ')
def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]
def finding_certain_model_regression(dataset):
    data = dataset
    print(missing_values_table(data))
    print(get_single_value_columns(data))

    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    for i in sorted(TEST_DF.columns):
        #['Season','PlayAttempted','ScoreDiff','PosTeamScore','FieldGoalDistance','DefTeamScore']
        df_dropped = drop_categorical_columns(data)
        #print(get_single_value_columns(df_dropped))
        #print(missing_values_table(df_dropped))
        label = i
        df=drop_label_with_null(df_dropped,label)
        num_rows_with_missing_values = df.isnull().any(axis=1).sum()


        X, y = split_features_labels(df, label)

        # Split the data into training and testing sets (adjust the test_size as needed)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Ensure X_train and X_test have the same columns
        common_columns = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]

        # Merge X_train and y_train into df_train
        df_train = pd.concat([X_train, Y_train], axis=1)

        # Merge X_test and y_test into df_test
        df_test = pd.concat([X_test, Y_test], axis=1)

        # Reset the index for df_train and df_test
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        missing_train, C_train = get_submatrix(df_train)
        missing_test, C_test = get_submatrix(df_test)
        # Ensure missing_train and missing_test have the same columns
        common_missing_columns = missing_train.columns.intersection(missing_test.columns)
        missing_train = missing_train[common_missing_columns]
        missing_test = missing_test[common_missing_columns]
        common_missing_columns = C_train.columns.intersection(C_test.columns)
        C_train = C_train[common_missing_columns]
        C_test = C_test[common_missing_columns]
        print(f"Label : {label}, Unique values: {len(df_dropped[label].unique())}, Unique values in Df_train: {len(df[label].unique())}")
        print(f"Original shape : {df_dropped.shape}, removing null {label} only: X_train: {df_train.shape}, Complete data: {C_train.shape}")


        CX_train, Cy_train = get_Xy(C_train, label)
        CX_test, Cy_test = get_Xy(C_test, label)
        CX_test = CX_test.reset_index(drop=True)
        Cy_test = Cy_test.reset_index(drop=True)

        start_time = time.time()
        result, CM_score = check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test)
        end_time = time.time()
        CM_time=end_time-start_time
        print(f'Certain Model result for {label}, time :{CM_time} Score {CM_score} and RESULT ###### {result} ')

        ###################
        n_samples = df.shape[0]
        n_features = df.shape[1]
        missing_factor = num_rows_with_missing_values / n_samples
        ###################
        # df, X_clean, y_clean, X_dirty, y_dirty, X_full, indices_dirty, indices_clean, size,train_indices, test_indices = generate_mock_data(df_train, df_test, i)
        # clean_test_indices = translate_indices(test_indices, indices_clean)
        # start_time = time.time()
        # #dirty_data, clean_data, test_data, full_data, indextuple,
        # AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty), (X_clean, y_clean),
        #                                        (X_clean[clean_test_indices, :], y_clean[clean_test_indices]), X_full,
        #                                        (train_indices, indices_dirty, indices_clean))
        # end_time = time.time()
        # ####################
        # features_test, target_test, X_clean, y_clean, X_dirty, y_dirty, X_full, train_indices, indices_dirty, indices_clean = genreate_AC_data(
        #     df_train, df_test, i)
        #
        # start_time = time.time()
        # AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
        #                                        (X_clean, y_clean),
        #                                        (features_test, target_test),
        #                                        X_full,
        #                                        (train_indices, indices_dirty, indices_clean))
        # end_time=time.time()
        # AC_records = (AC_records_1 ) / 1
        # AC_score = (AC_score_1 ) / 1
        # time_elapsed = end_time - start_time
        # AC_time = time_elapsed / 1
        #
        # print(f'N_samples: {n_samples} N_features: {n_features} N_missing_samples:{num_rows_with_missing_values} missing_factor: {missing_factor}')
        # print(f'Active Clean result for {label}, time :{AC_time} and Score ###### {AC_records} ')



if __name__ == '__main__':
    dataset = oml.datasets.get_dataset(43181)
    eeg, *_ =dataset.get_data()
    path = os.path.join('Final-Datasets', 'HMEQ-processed.csv')
    eeg=pd.read_csv(path)
    finding_certain_model_classification(eeg)