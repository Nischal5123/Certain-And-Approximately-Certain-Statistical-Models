import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
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
        svm_model = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)

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