import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
import os
import numpy as np
from utils import split_features_labels, missing_values_table
from scipy.sparse import csr_matrix
from ActiveClean import activeclean
from sklearn.preprocessing import MinMaxScaler
from utils import get_simple_imputer_model_regression
from utils import get_naive_imputer_model_regression,get_knn_imputer_model_regression,get_miwei_imputer_model_regression
from sklearn.metrics import mean_squared_error

def check_certain_model_regression(CX_train, Cy_train, Full_train_X, Full_train_y, missing_train, CX_test, Cy_test):
    #make sure order of columns in test and train is the same
    #assert(len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns)) #dont care about this if not getting accuracy
    reg = LinearRegression(fit_intercept = False).fit(CX_train.values,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_train.values,w_bar.T) - Cy_train)
    result = check_orthogonal(missing_train,loss)
    score=0.00000000001
    if result:
        clf = LinearRegression(fit_intercept=False).fit( Full_train_X.values, Full_train_y)
        y_pred = clf.predict(CX_test.values)
        score = mean_squared_error(y_pred, Cy_test.values)
    print(f"The mean squared error of the optimal model is {score:.2f}")
    return result,score

def get_submatrix(data):
    columns_without_nulls = data.columns[data.notnull().all()]
    C = data[columns_without_nulls]
    missing = data.drop(columns_without_nulls,axis = 1)
    return missing,C

# def get_submatrix(data):
#     rows_without_nulls = data.dropna()
#     C = data.loc[rows_without_nulls.index]
#     missing = data.drop(index=rows_without_nulls.index)
#     return missing, C

def get_Xy(data,label):
    X = data.drop(label,axis = 1)
    y = data[label]
    return X,y


def check_orthogonal(M,l):
    flag = True
    case = ''
    for j in range(len(l)):
        total = 0
        for i in range(M.shape[1]):
            print(i,j)
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


def active_clean_driver(df_train, df_test,label):
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
    ) = genreate_AC_data(df_train, df_test,label)
    task='regression'
    start_time = time.time()
    AC_records_1, AC_score_1 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),task
    )
    AC_records_2, AC_score_2 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),task
    )
    AC_records_3, AC_score_3 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),task
    )
    AC_records_4, AC_score_4 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),task
    )
    AC_records_5, AC_score_5 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),task
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

def drop_label_with_null(df, column_name):
    # Drop rows where the specified column is null
    df_cleaned = df.dropna(subset=[column_name])

    return df_cleaned

def finding_certain_model_regression(path):
        data = pd.read_csv(path, index_col=0)
        label='T'
        print(f"Label : {label}, Unique values: {len(data[label].unique())}")
        data=drop_label_with_null(data.copy(),label)
        print(missing_values_table(data))

        X, y =get_Xy(data.copy(), label)



        # Split the data into training and testing sets (adjust the test_size as needed)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        # Instantiate the MinMaxScaler
        # Count the number of missing values in each row
        missing_values_per_row = X_train.isnull().sum(axis=1)

        # Count the total number of rows with missing values
        rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])

        # Display the result
        print("Number of rows with missing values:", rows_with_missing_values)

        # Calculate the total number of examples
        total_examples = len(X_train)

        # Calculate the missing factor for each column
        missing_factor = rows_with_missing_values / total_examples
        print(f"Total example {X_train.shape}, MISSING FACTOR : {missing_factor}")
        scaler = MinMaxScaler()

        # Scale the training data
        X_train.loc[:, X_train.columns] = scaler.fit_transform(X_train[X_train.copy().columns])

        # Scale the test data
        X_test.loc[:, X_train.columns] = scaler.transform(X_test[X_train.copy().columns])


        # Merge X_train and y_train into df_train
        df_train = pd.concat([X_train, Y_train], axis=1)

        # Merge X_test and y_test into df_test
        df_test = pd.concat([X_test, Y_test], axis=1)
        df_test.dropna(inplace=True)
        assert (df_test.isnull().any().any()==False)

        # Reset the index for df_train and df_test
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        missing_train, C_train = get_submatrix(df_train)
        CX_train, Cy_train = get_Xy(C_train, label)

        #this is not required since we make sure the test data is already Clean but not for debugging
        missing_test, C_test = get_submatrix(df_test)

        CX_test, Cy_test = get_Xy(C_test, label)


        CX_test = CX_test.reset_index(drop=True)
        Cy_test = Cy_test.reset_index(drop=True)
        CX_train = CX_train.reset_index(drop=True)
        Cy_train = Cy_train.reset_index(drop=True)

        # assert (len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns))




        #there exist some missing data
        if C_train.shape != data.shape:
            print(f"Label : {label}, Unique values: {len(df_train[label].unique())}, Unique values in Df_train: {len(df_train[label].unique())}")

            #there is some non-missing data
            if CX_train.shape != missing_train.shape:
                start_time = time.time()
                result, CM_score = check_certain_model_regression(CX_train, Cy_train, X_train.fillna(0), Y_train, missing_train, CX_test, Cy_test)
                end_time = time.time()
                CM_time = end_time - start_time
                name = f"Air_Quality_CM_Exist_{result}.txt"
                filename = os.path.join('Final-Results', name)
                with open(filename, "w+") as file:
                    file.write(f"Number of Rows with missing values:{rows_with_missing_values}\n")
                    file.write(f"Missing Factor:{missing_factor}\n")
                    file.write(f"Running Time (CM):  {CM_time}\n")
                    file.write(f"Accuracy (CM):  {CM_score}\n")

                meiwei_imputer_score, meiwei_imputer_time = get_miwei_imputer_model_regression(df_train, df_test, label)

                knn_imputer_score, knn_imputer_time = get_knn_imputer_model_regression(df_train, df_test, label)

                simple_imputer_score, simpler_imputer_time = get_simple_imputer_model_regression(df_train, df_test,
                                                                                                 label)

                naive_imputer_score, naive_imputer_time = get_naive_imputer_model_regression(df_train, df_test, label)

                AC_records, AC_score, AC_time = active_clean_driver(df_train, df_test, label)
                with open(filename, "a+") as file:
                    file.write(f"Accuracy (Meiewi): {meiwei_imputer_score}\n")
                    file.write(f"Running Time (Meiewi):  {meiwei_imputer_time}\n")
                    file.write(f"Accuracy (KNN): {knn_imputer_score}\n")
                    file.write(f"Running Time (KNN):  {knn_imputer_time}\n")
                    file.write(f"Accuracy (MI): {simple_imputer_score}\n")
                    file.write(f"Running Time (MI):  {simpler_imputer_time}\n")
                    file.write(f"Accuracy (NI): {naive_imputer_score}\n")
                    file.write(f"Running Time (NI):  {naive_imputer_time}\n")
                    file.write(f"Number of Examples Cleaned (AC): {AC_records}\n")
                    file.write(f"Running Time (AC):  {AC_time}\n")
                    file.write(f"Accuracy (AC):  {AC_score}\n")


if __name__ == '__main__':
    path = os.path.join('Final-Datasets', 'Air-Quality.csv')
    finding_certain_model_regression(path)