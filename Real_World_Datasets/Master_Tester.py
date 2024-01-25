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
from sklearn.preprocessing import StandardScaler

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


def check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test):
    len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns)
    #reg = SGDRegressor(penalty = None).fit(CX,Cy)
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_train,w_bar.T) - Cy_train)
    result = check_orthogonal(missing_train,loss)
    # print('w_bar',w_bar)
    # print('loss',loss)
    #lm = sm.OLS(Cy_train, CX_train).fit()  # Running the linear model
    #print(lm.summary())
    return result, reg.score(CX_test,Cy_test)

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
            if np.isnan(M.iloc[j,i]) and not np.isclose(l[j], 0,atol=1e-02):
                flag = False
                case = 'case1: ' + str(l[j])
                break
            elif not np.isnan(M.iloc[j,i]):
                #print(f'inside case2 : M:{M.iloc[j,i]}, l:{l[j]}')
                total += M.iloc[j,i] * l[j]
        if not np.isclose(total ,0, atol = 1e-02):
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

    print(get_single_value_columns(data))

    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    for label in sorted(TEST_DF.columns):
        if label not in get_single_value_columns(TEST_DF):
            df_dropped = drop_categorical_columns(dataset)
            df = drop_label_with_null(df_dropped, label)
            del df_dropped
            # print(missing_values_table(df))
            #print(f'##########################     Number of rows with missing values {df.isnull().any(axis=1).sum()} ##########################')
            X, y = split_features_labels(df, label,'classification')

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
            # Instantiate the MinMaxScaler
            scaler = StandardScaler()

            # Scale the training data
            X_train = scaler.fit_transform(X_train)

            # Scale the test data
            X_test = scaler.transform(X_test)
            try:

                result, CM_score = check_certain_model_classification(X_train,Y_train,X_test,Y_test)

            except ValueError as e:
                print(e)
                result=False
                CM_score=None
            end_time = time.time()

            print(f'########################## Certain Model result for {label}, time :{end_time - start_time} , test accuracy {CM_score} and RESULT ###### {result} ########################## ')

def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]
def finding_certain_model_regression(dataset):
    data = dataset
    TEST_DF = drop_categorical_columns(data)
    for label in sorted(['critical_staffing_shortage_today_no']):

        df_dropped = drop_categorical_columns(dataset,featurize=True)
        df=drop_label_with_null(df_dropped,label)
        num_rows_with_missing_values = df.isnull().any(axis=1).sum()

        Whole_missing_df, Whole_clean_df=get_submatrix(df)
        X, y = split_features_labels(Whole_clean_df, label)
        miss_X=Whole_missing_df


        # Split the data into training and testing sets (adjust the test_size as needed)
        CX_train, CX_test, CY_train, CY_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Merge X_train and y_train into df_train
        C_df_train = pd.concat([CX_train, CY_train], axis=1)
        # Merge X_test and y_test into df_test
        C_df_test = pd.concat([CX_test, CY_test], axis=1)
        # Reset the index for df_train and df_test
        C_df_train.reset_index(drop=True, inplace=True)
        C_df_test.reset_index(drop=True, inplace=True)

        # Split the data into training and testing sets (adjust the test_size as needed)
        MX_train, MX_test = train_test_split(miss_X, test_size=0.2, random_state=42)
        # Merge X_train and y_train into df_train
        M_df_train = MX_train
        # Merge X_test and y_test into df_test
        M_df_test = MX_test
        # Reset the index for df_train and df_test
        M_df_train.reset_index(drop=True, inplace=True)
        M_df_test.reset_index(drop=True, inplace=True)

        if Whole_clean_df.shape != df.shape:
            print(f"Label : {label}, Unique values: {len(df_dropped[label].unique())}, Unique values in Df_train: {len(df[label].unique())}")
            #print(f"Categorical dropped : {df_dropped.shape}, removing null {label} only:{df.shape} X_train: {df_train.shape}, Complete data: {C_train.shape}")


            CX_train, Cy_train = get_Xy(C_df_train, label)
            CX_train = CX_train.reset_index(drop=True)
            Cy_train = Cy_train.reset_index(drop=True)


            CX_test, Cy_test = get_Xy(C_df_test, label)
            CX_test = CX_test.reset_index(drop=True)
            Cy_test = Cy_test.reset_index(drop=True)



            start_time = time.time()

            #if failed for 1 attribute keep going
            # Reasons to fail : 1) got only 1 label 2) empty complete data
            if CX_train.shape != M_df_train.shape:
                try:
                    result, CM_score = check_certain_model_regression(CX_train, Cy_train, M_df_train, CX_test, Cy_test, M_df_test)
                except ValueError as e:
                    print(e)
                    result=False
                    CM_score=-100
                end_time = time.time()
                CM_time=end_time-start_time
                if result==True:
                    print(f'Certain Model result for {label}, time :{CM_time} Score {CM_score} and RESULT ###### {result} ')




if __name__ == '__main__':
    data_path = 'TestDataset/'
    errorfile=[]
    successfile=[]
    nonullfiles=[]
    csv_files = os.listdir(data_path)
    for csv_filename in sorted(csv_files):
        #'fraudfull.csv','city_day.csv','MELBOURNE_HOUSE_PRICES_LESS.csv','nflplaybyplay2015.csv','NFL.csv','uniprot_1001r_223c.csv','Tax.csv',
        if csv_filename.endswith('.csv') and csv_filename not in ['classification.csv']:
            path = os.path.join(data_path, csv_filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(path)
            #print(missing_values_table(df))
            # df.drop(columns=['geocoded_state'],inplace=True)
            # columns_to_drop = [
            #     'previous_day_admission_pediatric_covid_confirmed_0_4_coverage',
            #     'previous_day_admission_pediatric_covid_confirmed_12_17',
            #     'previous_day_admission_pediatric_covid_confirmed_12_17_coverage',
            #     'previous_day_admission_pediatric_covid_confirmed_5_11',
            #     'previous_day_admission_pediatric_covid_confirmed_5_11_coverage',
            #     'previous_day_admission_pediatric_covid_confirmed_unknown',
            #     'previous_day_admission_pediatric_covid_confirmed_unknown_coverage'
            # ]
            #
            #columns_to_drop=['292','293','157','158']
            #df.drop(columns=columns_to_drop,inplace=True)
            #
            # # %%
            # # Convert date and time columns to datetime format
            # df['date'] = pd.to_datetime(df['date'])
            #
            # # Extract year, month, day from the date column
            # df['year'] = df['date'].dt.year
            # df['month'] = df['date'].dt.month
            # df['day'] = df['date'].dt.day
            #
            # # Drop the original date and time columns
            # df = df.drop(['date'], axis=1)
            # # Generate column names as strings '1', '2', '3', ..., 'n' where n is the number of columns
            column_names = [str(i) for i in range(1, len(df.columns) + 1)]
            #
            # # Assign the generated column names to the DataFrame
            #df.columns = column_names
            # Replace '?' with NaN in the entire DataFrameN
            df.replace('?', np.nan, inplace=True)
            # df.replace('', np.nan, inplace=True)

            # nan_indices = np.random.choice(df.index, size=1, replace=False)
            # df.loc[nan_indices, '439'] = np.nan

            #print("Converting '{}'...".format(csv_filename))

            if df.isnull().values.any():

                    missing_values_per_row = df.isnull().sum(axis=1)

                    # Count the total number of rows with missing values
                    rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])
                    # Display the result
                    #print("Number of rows with missing values:", rows_with_missing_values)

                    # Calculate the missing factor for each column
                    missing_factor = rows_with_missing_values / df.shape[0]
                    print(f" Dataset {csv_filename}, Total example {df.shape}, MISSING FACTOR : {missing_factor*100}")
                    #finding_certain_model_regression(df)
                    successfile.append(csv_filename)

            else:
                nonullfiles.append(csv_filename)
                #print('No-null')
    print(f'Succesfully processes file:{successfile}, Failed files:{errorfile}, No Nullfiles:{len(nonullfiles)}')