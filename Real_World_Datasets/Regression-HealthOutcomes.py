import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import missing_values_table
from utils import drop_categorical_columns
from utils import featurize_categorical_data
import time
def check_certain_model(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test):
    #reg = SGDRegressor(penalty = None).fit(CX,Cy)
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_test,w_bar.T) - Cy_test)
    result = check_orthogonal(missing_test,loss)
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

def split_features_labels(df, label_column):
    # Check if the specified label column exists in the DataFrame
    if label_column not in df.columns:
        print(f"Label column '{label_column}' not found in the DataFrame.")
        return None, None
    df_drop_label= df
    # Features (X) are all columns except the specified label column
    X = df_drop_label.drop(label_column, axis=1)

    # Label (y) is the specified column
    # Create a new binary column based on the specified midpoint
    if len(df_drop_label[label_column].unique())!=2:
        midpoint = df_drop_label[label_column].mean()
        df_drop_label[label_column] = df_drop_label[label_column].apply(lambda x: 1 if x > midpoint else 0)

    y = df_drop_label[label_column]

    return X, y

if __name__ == '__main__':
    data = pd.read_csv("/Users/aryal/Desktop/Personal/Certain-Statistical-Models/Real_World_Datasets/data/nflplaybyplay2015.csv")
    print(missing_values_table(data))
    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    print(missing_values_table(TEST_DF))
    for i in ['2015']:
        df = drop_categorical_columns(data)
        print(missing_values_table(df))
        label=i

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
        print(f"Original shape: X_train: {df_train.shape}, Shape after removal: {C_train.shape}")

        CX_train, Cy_train = get_Xy(C_train, label)

        missing_test, C_test = get_submatrix(df_test)
        CX_test, Cy_test = get_Xy(C_test, label)
        CX_test = CX_test.reset_index(drop=True)
        Cy_test = Cy_test.reset_index(drop=True)

        start_time = time.time()
        result, CM_score = check_certain_model(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test)
        end_time = time.time()
        print(f'Certain Model result for {label}, time :{end_time-start_time} and RESULT ###### {result} ')
