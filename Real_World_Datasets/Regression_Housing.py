import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
import os
import numpy as np



def drop_label_with_null(df, column_name):
    # Drop rows where the specified column is null
    df_cleaned = df.dropna(subset=[column_name])

    return df_cleaned
def drop_categorical_columns(df):
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    # Drop categorical columns from the DataFrame
    df_without_categorical =  df.drop(categorical_columns, axis=1)

    return df_without_categorical

def check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test):
    #reg = SGDRegressor(penalty = None).fit(CX,Cy)
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_train,w_bar.T) - Cy_train)
    result = check_orthogonal(missing_train,loss)
    # print('w_bar',w_bar)
    # print('loss',loss)
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


def finding_certain_model_regression(path):

    data = pd.read_csv(path,index_col=0)
    df_dropped = drop_categorical_columns(data)  # 1) only considering numerical features
    label = 'SalePrice'
    df=drop_label_with_null(df_dropped,label)  # 2) label cannot contain null values

    X, y = get_Xy(df, label)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merge X_train and y_train into df_train
    df_train = pd.concat([X_train, Y_train], axis=1)
    # Merge X_test and y_test into df_test
    df_test = pd.concat([X_test, Y_test], axis=1)
    # Reset the index for df_train and df_test
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    missing_train, C_train = get_submatrix(df_train)
    print(f"Label : {label}, Unique values: {len(df_dropped[label].unique())}, Unique values in Df_train: {len(df[label].unique())}")
    print(f"Original shape : {df_dropped.shape}, removing null {label} only: X_train: {df_train.shape}, Complete data: {C_train.shape}")


    CX_train, Cy_train = get_Xy(C_train, label)

    missing_test, C_test = get_submatrix(df_test)
    CX_test, Cy_test = get_Xy(C_test, label)
    CX_test = CX_test.reset_index(drop=True)
    start_time = time.time()
    result, CM_score = check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test)
    end_time = time.time()
    CM_time=end_time-start_time
    print(f'Certain Model result for {label}, time :{CM_time}  RESULT ###### {result} and score {CM_score}')



if __name__ == '__main__':
    path = os.path.join('data', 'house-prices-advanced-regression-techniques/train.csv')
    finding_certain_model_regression(path)