import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from utils import split_features_labels
import statsmodels.api as sm
import pickle

def drop_label_with_null(df, column_name):
    # Drop rows where the specified column is null
    df_cleaned = df.dropna(subset=[column_name])

    return df_cleaned

#['time_missing', 'SideofField_missing', 'posteam_missing', 'DefensiveTeam_missing', 'ExPointResult_missing',
    # 'TwoPointConv_missing', 'DefTwoPoint_missing', 'PuntResult_missing', 'Passer_missing',
    # 'PassOutcome_missing', 'PassLength_missing', 'PassLocation_missing', 'Interceptor_missing',
    # 'Rusher_missing', 'RunLocation_missing', 'RunGap_missing', 'Receiver_missing', 'ReturnResult_missing',
    # 'Returner_missing', 'BlockingPlayer_missing', 'Tackler1_missing', 'Tackler2_missing',
    # 'FieldGoalResult_missing', 'RecFumbTeam_missing', 'RecFumbPlayer_missing', 'ChalReplayResult_missing',
    # 'PenalizedTeam_missing', 'PenaltyType_missing', 'PenalizedPlayer_missing']

def manual_categorical_imputation(df, categorical_columns):
    df=df.reset_index(drop=True)
     # Step 1: Fill categorical missings with "missing"
    df[categorical_columns] = df[categorical_columns].fillna("missing")
    assertion = df.isin(['missing']).any().any()
    # Step 2: Use OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    encoded_data = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
    assertion = encoded_data.isin(['missing']).any().any()
    # Step 3: Get feature names and identify missing indicator columns
    feature_names = encoder.get_feature_names_out()
    missing_indicator_cols = [col for col in feature_names if '_missing' in col]
    #df = pd.concat([df, encoded_data], axis=1)

    # Step 4: Replace original categorical columns with NaN where missing indicator is 1
    for categorical_col in categorical_columns:
        missing_indicator_col = f"{categorical_col}_missing"

        if missing_indicator_col in missing_indicator_cols:
            mask = (encoded_data[missing_indicator_col] == 1)

            # Replace all columns that start with categorical_col with NaN where missing indicator is 1
            cols_to_replace = [col for col in encoded_data.columns if col.startswith(categorical_col)]
            encoded_data.loc[mask, cols_to_replace] = np.nan

    encoded_data.drop(columns=missing_indicator_cols)
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_data], axis=1)

    return df
def drop_categorical_columns(df,conversion=False,featurize=False):
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object','category']).columns.tolist()
    return_df = df

    if conversion==True:
        for col in categorical_columns:
            # Find the most common value in the column
            most_common_value = df[col].mode().iloc[0]

            # Map the non-null column values accordingly
            df[col] = df[col].apply(
                lambda x: 1 if pd.notna(x) and x == most_common_value else (0 if pd.notna(x) else x))

        return_df=df.copy()

    elif featurize==True:
        return_df=manual_categorical_imputation(df,categorical_columns)

    else:
        #this is to avoid droping int and float mixed type columns since they will be considered objects
        for col in categorical_columns:
                df[col]=pd.to_numeric(df[col], errors='ignore')
        #after those are taken care of we can drop the columns that are still object
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return_df = df.drop(categorical_columns, axis=1)

        # # Drop categorical columns from the DataFrame
        # return_df =  df.drop(categorical_columns, axis=1)

    return return_df

def check_certain_model_regression(CX_train, Cy_train, missing_train, CX_test, Cy_test, missing_test):
    #make sure order of columns in test and train is the same
    assert(len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns))
    reg = LinearRegression(fit_intercept = False).fit(CX_train,Cy_train)
    w_bar = reg.coef_
    loss = (np.dot(CX_train,w_bar.T) - Cy_train)
    result = check_orthogonal(missing_train,loss)
    # print('w_bar',w_bar)
    # print('loss',loss)
    y_pred = reg.predict(CX_test)
    plt.scatter(Cy_test, y_pred, color='k')
    plt.show()

    X_train_rfe = sm.add_constant(CX_train)
    lm = sm.OLS(Cy_train,CX_train).fit()   # Running the linear model
    print(lm.summary())
    return result,reg.score(CX_test,Cy_test)

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

def create_numpy_file(path):
    data = pd.read_csv(path, index_col=0)
    data = data.drop(columns=['Date'])
    data = drop_categorical_columns(data)  # 1) only considering numerical features
    del data
def finding_certain_model_regression(path,pickled=False):
    if pickled:
        full_path=os.path.join(path, 'nfl.pkl')
        # Load NumPy array from pickle
        with open(full_path, 'rb') as file:
            loaded_array, loaded_columns = pickle.load(file)
        # Convert NumPy array back to DataFrame
        data = pd.DataFrame(loaded_array,columns=loaded_columns)[:100]
        del loaded_columns
        del loaded_array
    else:
        data = pd.read_csv(path, index_col=0)
        data.replace('?', np.nan, inplace=True)
        #data=data.drop(columns=['Date','time','desc','PlayType'])
        data = drop_categorical_columns(data,conversion=True)
    for label in data.columns:
        data=drop_label_with_null(data,label)  # 2) label cannot contain null values
        print(f"Label : {label}, Unique values: {len(data[label].unique())}")


        Whole_missing_df, Whole_clean_df=get_submatrix(data)
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

        if Whole_clean_df.shape != data.shape:
            #print(f"Label : {label}, Unique values: {len(df_dropped[label].unique())}, Unique values in Df_train: {len(df[label].unique())}")
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
            df_train_C=pd.concat([CX_train,Cy_train],axis=1)
            df_train=pd.concat([df_train_M_df_test],axis=0)
            features_test, target_test, X_clean, y_clean, X_dirty, y_dirty, X_full, train_indices, indices_dirty, indices_clean = genreate_AC_data(
                df_train, df_test, 'ScoreDiff')

            start_time = time.time()
            AC_records, AC_score = activeclean((X_dirty, y_dirty),
                                               (X_clean, y_clean),
                                               (features_test, target_test),
                                               X_full,
                                               (train_indices, indices_dirty, indices_clean), 'regression')
            end_time = time.time()

            # Calculate the elapsed time
            ac_elapsed_time = end_time - start_time
            print(
                f'########################## Records_Cleaned {AC_records}, time :{ac_elapsed_time} , test accuracy {AC_score} ')





if __name__ == '__main__':
    path = os.path.join('TestDataset', 'ncvoter.csv')
    #create_numpy_file(path)
    finding_certain_model_regression(path)