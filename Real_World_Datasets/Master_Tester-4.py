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


    #print(f"Original shape: X_train: {X_train.shape}, X_train: {X_train_complete.shape}")
    # print(f"Shape after removal: X_train: {X_train_complete.shape}, y_train: {y_train_complete.shape}")
    if X_train.shape==X_train_complete.shape:
        return False , None


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
        if abs(feature_weights[i]) >= 1e-01:
            res = False
            print('Case 1')
            return res, None

            # Return False as soon as a condition is not met

    # Check the condition for all rows in X_train_missing_rows
    for i in range(len(X_train_missing_rows)):
        row = X_train_missing_rows[i]
        label = y_train_missing_rows[i]
        dot_product = np.sum(row[~np.isnan(row)] * feature_weights[~np.isnan(row)])
        if label * dot_product <= 1:
            res = False
            print('Case 2')
            return res, None
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

def finding_certain_model_classification(dataset,file):
    data = dataset

    print(get_single_value_columns(data))
    print(missing_values_table(data))
    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    for label in sorted(TEST_DF.columns):
        max_attempts=1000
        if label not in get_single_value_columns(TEST_DF):

            for situation in ['mean']:
                while max_attempts > 0:
                    print(f"Converting {file}, label {label}, situtation {situation}, attempt {max_attempts}")

                    df_dropped = drop_categorical_columns(dataset)
                    df = drop_label_with_null(df_dropped, label)
                    del df_dropped
                    missing_values_table(df)
                    #print(f'##########################     Number of rows with missing values {df.isnull().any(axis=1).sum()} ##########################')
                    X, y = split_features_labels(df, label,'classification',situation)

                    # Split the data into training and testing sets (adjust the test_size as needed)
                    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Merge X_train and y_train into df_train
                    df_train = pd.concat([X_train, Y_train], axis=1)
                    random_seed = np.random.randint(1, 10000)

                    # Merge X_test and y_test into df_test
                    df_test = pd.concat([X_test, Y_test], axis=1)

                    # Reset the index for df_train and df_test
                    df_train.reset_index(drop=True, inplace=True)
                    df_train = make_dirty(df_train, random_seed, 0.001)
                    df_train.reset_index(drop=True, inplace=True)

                    df_test.reset_index(drop=True, inplace=True)


                    missing_train, C_train = get_submatrix(df_train)
                    CX_train, Cy_train = get_Xy(C_train, label)

                    missing_test, C_test = get_submatrix(df_test)
                    CX_test, Cy_test = get_Xy(C_test, label)
                    CX_test = CX_test.reset_index(drop=True)
                    Cy_test = Cy_test.reset_index(drop=True)


                    # Instantiate the MinMaxScaler
                    scaler = MinMaxScaler()

                    # Scale the training data
                    X_train = scaler.fit_transform(X_train)

                    # Scale the test data
                    X_test = scaler.transform(X_test)
                    max_attempts-=1
                    start_time = time.time()
                    try:

                        result, CM_score = check_certain_model_classification(X_train,Y_train,X_test,Y_test)

                    except ValueError as e:
                        print(e)
                        result=False
                        CM_score=0
                    end_time = time.time()
                    with open('Master-Tester.txt', 'a+') as writefile:
                        writefile.write(f'###### {file} #################### Certain Model result for {label}, time :{end_time - start_time} , test accuracy {CM_score} and RESULT ###### {result}, situation {situation}, random_seed {random_seed} ##########################\n')

                    print(
                        f'########################## Certain Model result for {label}, time :{end_time - start_time} , test accuracy {CM_score} and RESULT ###### {result}, situation {situation}, random_seed {random_seed} ##########################\n')
                    if result:
                        print(f'########################## Certain Model result for {label}, time :{end_time - start_time} , test accuracy {CM_score} and RESULT ###### {result}, situation {situation}, random_seed {random_seed} ##########################\n')
                        break


if __name__ == '__main__':
    data_path = 'TestDataset/'
    errorfile=[]
    successfile=[]
    nonullfiles=[]
    csv_files = os.listdir(data_path)
    for csv_filename in sorted(csv_files):
        #'fraudfull.csv','city_day.csv','MELBOURNE_HOUSE_PRICES_LESS.csv','nflplaybyplay2015.csv','NFL.csv','uniprot_1001r_223c.csv','Tax.csv',
        if csv_filename.endswith('tuandromd.csv') and csv_filename not in ['CommunitiesAndCrime.csv','survey.csv','uniprot_1001r_223c.csv','tsa_claims3_11.csv','classification.csv']:
            path = os.path.join(data_path, csv_filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path,header=None)[1:]

            # # Generate column names as strings '1', '2', '3', ..., 'n' where n is the number of columns
            column_names = [str(i) for i in range(1, len(df.columns) + 1)]

            # # Assign the generated column names to the DataFrame
            df.columns = column_names
            # Replace '?' with NaN in the entire DataFrameN
            print(missing_values_table(df))
            df.replace('?', np.nan, inplace=True)
            print(missing_values_table(df))

            # # Assign the generated column names to the DataFrame
            df.columns = column_names
            # df.replace('', np.nan, inplace=True)
            if df.isnull().values.any():
                    print("Converting '{}'...".format(csv_filename))
                    finding_certain_model_classification(df,csv_filename)
                    successfile.append(csv_filename)
                    errorfile.append(csv_filename)
            else:
                nonullfiles.append(csv_filename)
                #print('No-null')
    print(f'Succesfully processes file:{successfile}, Failed files:{errorfile}, No Nullfiles:{len(nonullfiles)}')