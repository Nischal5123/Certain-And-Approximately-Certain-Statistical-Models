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
from CertainModel import get_Xy,get_submatrix,check_certain_model_classification,check_certain_model_regression




def finding_certain_model_classification(path):
    data = pd.read_csv(path)
    print(missing_values_table(data))
    print(get_single_value_columns(data))

    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    # print(missing_values_table(TEST_DF))
    for i in TEST_DF.columns:
        df_dropped = drop_categorical_columns(data)
        if i not in get_single_value_columns(df_dropped):
            # print(missing_values_table(df_dropped))
            label = i
            df = drop_label_with_null(df_dropped, label)

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
            print(f'Certain Model result for {label}, time :{end_time - start_time} and RESULT ###### {result} ')

def finding_certain_model_regression(path):
    data = pd.read_csv(path)
    print(missing_values_table(data))
    print(get_single_value_columns(data))

    # data = drop_categorical_columns(df)
    TEST_DF = drop_categorical_columns(data)
    #print(missing_values_table(TEST_DF))
    for i in sorted(['Season','PlayAttempted','ScoreDiff','PosTeamScore','FieldGoalDistance','DefTeamScore']):
        #['Season','PlayAttempted','ScoreDiff','PosTeamScore','FieldGoalDistance','DefTeamScore']
        df_dropped = drop_categorical_columns(data)
        #print(get_single_value_columns(df_dropped))
        #print(missing_values_table(df_dropped))
        label = i
        df=drop_label_with_null(df_dropped,label)

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
        print(f"Label : {label}, Unique values: {len(df_dropped[label].unique())}, Unique values in Df_train: {len(df[label].unique())}")
        print(f"Original shape : {df_dropped.shape}, removing null {label} only: X_train: {df_train.shape}, Complete data: {C_train.shape}")


        CX_train, Cy_train = get_Xy(C_train, label)

        missing_test, C_test = get_submatrix(df_test)
        CX_test, Cy_test = get_Xy(C_test, label)
        CX_test = CX_test.reset_index(drop=True)
        Cy_test = Cy_test.reset_index(drop=True)

        start_time = time.time()
        result, CM_score = check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test)
        end_time = time.time()
        CM_time=end_time-start_time
        print(f'Certain Model result for {label}, time :{CM_time} and RESULT ###### {result} ')



if __name__ == '__main__':
    path = os.path.join('data', 'nflheaders.csv')
    finding_certain_model_regression(path)