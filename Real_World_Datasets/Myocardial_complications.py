import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from utils import missing_values_table
from utils import drop_categorical_columns
from ucimlrepo import fetch_ucirepo
def check_certain_model(X_train, y_train):
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
    return res, feature_weights

def split_features_labels(df, label_column):
    # Check if the specified label column exists in the DataFrame
    if label_column not in df.columns:
        print(f"Label column '{label_column}' not found in the DataFrame.")
        return None, None
    df_new=df.dropna(subset=[label_column],axis=0)
    # Features (X) are all columns except the specified label column
    X = df_new.drop(label_column, axis=1)

    # Label (y) is the specified column
    # Create a new binary column based on the specified midpoint
    if len(df_new[label_column].unique())!=2:
        midpoint = df_new[label_column].mean()
        df_new[label_column] = df_new[label_column].apply(lambda x: 1 if x > midpoint else -1)

    y = df_new[label_column]

    return X, y

if __name__ == '__main__':

    myocardial_infarction_complications = fetch_ucirepo(id=579)

    # data (as pandas dataframes)
    X = myocardial_infarction_complications.data.features
    target = myocardial_infarction_complications.data.targets
    X=X.drop('KFK_BLOOD',axis=1)
    X=X.drop('IBS_NASL',axis=1)
    X = X.drop('D_AD_KBRIG', axis=1)
    X = X.drop('S_AD_KBRIG', axis=1)
    X = drop_categorical_columns(X)
    print(missing_values_table(X))


    for i in (X.columns):
        print('Checking column as label',i)
        label_column=i
        X,y=split_features_labels(X,i)
        print(missing_values_table(X))
        if label_column=='nr_07':
            print('stop')
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        res=False

        res,_=check_certain_model(X, y)
        if res==True:
                print('######################################      Found !!!! ##########################',i)
                break
        print(f'Certain model exists: {res}',i)
