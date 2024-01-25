import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
import os
import numpy as np
from utils import split_features_labels, missing_values_table,drop_label_with_null
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from utils import get_simple_imputer_model_regression
from utils import get_naive_imputer_model_regression
from sklearn.metrics import mean_squared_error,r2_score
from utils import drop_categorical_columns
import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]

def error_classifier(total_labels, full_data):
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices,:],labels)
        return clf
    else:
        return None

def ec_filter(dirtyex, full_data, clf, t=0.90):
    if clf != None:
        pred = clf.predict_proba(full_data[dirtyex,:])
        return [j for i,j in enumerate(dirtyex) if pred[i][0] < t]

    #print("CLF none")
    return dirtyex




def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, task='classification', batchsize=50, total=10000):
    # makes sure the initialization uses the training data
    X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
    y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0].values
    y_test = test_data[1].values

    # print("[ActiveClean Real] Initialization")

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    total_labels = []
    total_cleaning = 0  # Initialize the total count of missing or originally dirty examples

    ##Not in the paper but this initialization seems to work better, do a smarter initialization than
    ##just random sampling (use random initialization)
    topbatch = np.random.choice(range(0, len(dirtyex)), batchsize)
    examples_real = [dirtyex[j] for j in topbatch]
    examples_map = translate_indices(examples_real, indextuple[2])

    # Apply Cleaning to the Initial Batch

    cleanex.extend(examples_map)
    for j in set(examples_real):
        dirtyex.remove(j)

    if task =='classification':
        clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
        clf.fit(X_clean[cleanex, :], y_clean[cleanex])
    else:
        clf = SGDRegressor(penalty=None, alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
        clf.fit(X_clean[cleanex, :], y_clean[cleanex])

    for i in range(50, total, batchsize):
        # print("[ActiveClean Real] Number Cleaned So Far ", len(cleanex))
        ypred = clf.predict(X_test)
        # print("[ActiveClean Real] Prediction Freqs",np.sum(ypred), np.shape(ypred))
        # print(f"[ActiveClean Real] Prediction Freqs sum ypred: {np.sum(ypred)} shape of ypred: {np.shape(ypred)}")
        # print(classification_report(y_test, ypred))

        # Sample a new batch of data
        examples_real = np.random.choice(dirtyex, batchsize)

        # Calculate the count of missing or originally dirty examples within the batch
        missing_count = sum(1 for r in examples_real if r in indextuple[1])
        total_cleaning += missing_count  # Add the count to the running total

        examples_map = translate_indices(examples_real, indextuple[2])

        total_labels.extend([(r, (r in lset)) for r in examples_real])

        # on prev. cleaned data train error classifier
        ec = error_classifier(total_labels, full_data)

        for j in examples_real:
            try:
                dirtyex.remove(j)
            except ValueError:
                pass

        dirtyex = ec_filter(dirtyex, full_data, ec)

        # Add Clean Data to The Dataset
        cleanex.extend(examples_map)

        # uses partial fit (not in the paper--not exactly SGD)
        clf.partial_fit(X_clean[cleanex, :], y_clean[cleanex])

        print('Clean', len(cleanex))
        # print("[ActiveClean Real] Accuracy ", i ,accuracy_score(y_test, ypred,normalize = True))
        #print(f"[ActiveClean Real] Iteration: {i} Accuracy: {accuracy_score(y_test, ypred,normalize = True)}")
        if len(dirtyex) < 50:
            print("[ActiveClean Real] No More Dirty Data Detected")
            print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
            # return total_cleaning, 0 if clf.score(X_clean[cleanex, :], y_clean[cleanex]) is None else clf.score(
            #     X_clean[cleanex, :], y_clean[cleanex])cleanex
            if task == 'classification':
                ypred = clf.predict(X_test)
                return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

            else:
                #ypred = clf.predict(X_test)
                return total_cleaning, 0 if r2_score(y_test, ypred) is None else r2_score(y_test, ypred)

    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    if task == 'classification':
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

    else:
        #ypred = clf.predict(X_test)
        return total_cleaning, 0 if r2_score(y_test, ypred) is None else r2_score(y_test, ypred)

def check_certain_model_regression(CX_train, Cy_train, Full_train_X, Full_train_y, missing_train, CX_test, Cy_test):
    #make sure order of columns in test and train is the same
    #assert(len(CX_test.columns) == len(CX_train.columns) and all(CX_test.columns == CX_train.columns)) #dont care about this if not getting accuracy
    reg = LinearRegression(fit_intercept = False).fit(CX_train.values,Cy_train.values)
    w_bar = reg.coef_
    loss = (np.dot(CX_train.values,w_bar.T) - Cy_train.values)
    result = check_orthogonal(missing_train,loss)
    score=0.00000000001
    if result:
        clf = LinearRegression(fit_intercept=False).fit( Full_train_X.values, Full_train_y.values)
        y_pred = clf.predict(CX_test.values)
        score = mean_squared_error(y_pred, Cy_test.values)
    print(f"The mean squared error of the optimal model is {score:.2f}")
    return result,score

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
                print(case)
                break
            elif not np.isnan(M.iloc[j,i]):
                #print(f'inside case2 : M:{M.iloc[j,i]}, l:{l[j]}')
                total += M.iloc[j,i] * l[j]
        if not np.isclose(total ,0, atol = 1e-03):
            flag = False
            case = 'case2: ' + str(total)
            print(case)
            break
    print(case)
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
            e_feat[i, j] = np.random.rand()
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

    start_time = time.time()
    AC_records_1, AC_score_1 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),'regression'
    )
    AC_records_2, AC_score_2 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),'regression'
    )
    AC_records_3, AC_score_3 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),'regression'
    )
    AC_records_4, AC_score_4 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),'regression'
    )
    AC_records_5, AC_score_5 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),'regression'
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

def finding_certain_model_regression(path):
        df = pd.read_csv(path)
        label='Price'
        data=drop_label_with_null(df,label)
        print(missing_values_table(df.copy()))
        print(f"Label : {label}, Unique values: {len(data[label].unique())}")

        X, y = get_Xy(data, label)



        # Split the data into training and testing sets (adjust the test_size as needed)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        scaler2 = StandardScaler()
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


        CX_test, Cy_test = get_Xy(df_test, label)
        CX_test = CX_test.reset_index(drop=True)
        Cy_test = Cy_test.reset_index(drop=True)

        # assert (len(CX_test.columns) == len(X_train.columns) )





        if C_train.shape != data.shape:
            print(f"Label : {label}, Unique values: {len(df_train[label].unique())}, Unique values in Df_train: {len(df_train[label].unique())}")
            # print(f"Categorical dropped : {df_dropped.shape}, removing null {label} only:{df.shape} X_train: {df_train.shape}, Complete data: {C_train.shape}")


            if CX_train.shape != missing_train.shape:
                start_time = time.time()
                #X_train.fillna(0) is used to fit the model only when certain model exist. This is to make sure number of features match in test and train set
                #CX_train is complete subset (feature < original features) : without missing features. X_train is imputed version
                result, CM_score =  check_certain_model_regression(CX_train, Cy_train, X_train.fillna(0), Y_train, missing_train, CX_test, Cy_test)
                end_time = time.time()
                CM_time = end_time - start_time
                print('###################################',result)
                filename = f'Laptop_CM_Exist_{result}.txt'
                with open(filename, "w+") as file:
                    file.write(f"Number of Rows with missing values:{rows_with_missing_values}\n")
                    file.write(f"Missing Factor:{missing_factor}\n")
                    file.write(f"Running Time (CM):  {CM_time}\n")
                    file.write(f"Accuracy (CM):  {CM_score}\n")

                simple_imputer_score, simpler_imputer_time = get_simple_imputer_model_regression(df_train, df_test,label)


                naive_imputer_score, naive_imputer_time = get_naive_imputer_model_regression(df_train, df_test,label)

                AC_records, AC_score, AC_time = active_clean_driver(df_train, df_test,label)
                with open(filename, "a+") as file:
                        file.write(f"Accuracy (MI): {simple_imputer_score}\n")
                        file.write(f"Running Time (MI):  {simpler_imputer_time}\n")
                        file.write(f"Accuracy (NI): {naive_imputer_score}\n")
                        file.write(f"Running Time (NI):  {naive_imputer_time}\n")
                        file.write(f"Number of Examples Cleaned (AC): {AC_records}\n")
                        file.write(f"Running Time (AC):  {AC_time}\n")
                        file.write(f"Accuracy (AC):  {AC_score}\n")


if __name__ == '__main__':
    path = os.path.join('Final-Datasets', 'Laptop-encoded.csv')
    finding_certain_model_regression(path)