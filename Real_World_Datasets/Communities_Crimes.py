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
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test):
    #make sure order of columns in test and train is the same
    # assert(len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns)) #dont care about this if not getting accuracy
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_train,w_bar.T) - Cy_train)
    result = check_orthogonal(missing_train,loss)
    y_pred = reg.predict(CX_test)
    score = mean_squared_error(Cy_test, y_pred)
    print(f"The mean squared error of the optimal model is {score:.2f}")
    #score=reg.score(CX_test,Cy_test)
    return result,score

def get_submatrix(data):
    rows_without_nulls = data.dropna()
    C = data.loc[rows_without_nulls.index]
    missing = data.drop(index=rows_without_nulls.index)
    return missing, C

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

def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]
def finding_certain_model_regression(path):
    # fetch dataset
    data = pd.read_csv(path)
    print(missing_values_table(data))
    columns=['ViolentCrimesPerPop']
    for label in sorted(columns):
            df=drop_label_with_null(data.copy(),label)
            num_rows_with_missing_values = df.isnull().any(axis=1).sum()

            X, y = split_features_labels(df, label)

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
                result, CM_score = check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test)
            except AssertionError as e:
                print('############# Assertion Error ###########')
                result=False
                CM_score=-100000
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
            features_test, target_test, X_clean, y_clean, X_dirty, y_dirty, X_full, train_indices, indices_dirty, indices_clean = genreate_AC_data(
                df_train, df_test, i)

            start_time = time.time()
            AC_records_1, AC_score_1 = activeclean((X_dirty, y_dirty),
                                                   (X_clean, y_clean),
                                                   (features_test, target_test),
                                                   X_full,
                                                   (train_indices, indices_dirty, indices_clean))
            end_time=time.time()
            AC_records = (AC_records_1 ) / 1
            AC_score = (AC_score_1 ) / 1
            time_elapsed = end_time - start_time
            AC_time = time_elapsed / 1

            print(f'N_samples: {n_samples} N_features: {n_features} N_missing_samples:{num_rows_with_missing_values} missing_factor: {missing_factor}')
            print(f'Active Clean result for {label}, time :{AC_time} and Score ###### {AC_records} ')



if __name__ == '__main__':
    path = os.path.join('Final-Datasets', 'CommunitiesAndCrime.csv')
    finding_certain_model_regression(path)