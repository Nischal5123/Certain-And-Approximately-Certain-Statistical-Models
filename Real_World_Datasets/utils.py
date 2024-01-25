import pandas as pd
import missingno as msno
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
import scipy.sparse
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import time
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
#from hyperimpute.plugins.imputers import Imputers
# import MIDASpy as md
# import datawig


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Data types of columns
    data_types = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, data_types], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Data Type'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def tryParse(X):
    vals = []

    if X.shape == (1, 1):
        try:
            vals.append(float(X.tolist()[0][0]))
        except ValueError:
            vals.append(0)

        return vals

    for x in np.squeeze(X.T):
        try:
            vals.append(float(x))
        except ValueError:
            vals.append(0)

    return vals

# def featurize_dataframe(df):
#     feature_list = []
#     transform_list = []
#
#     # Infer types based on DataFrame dtypes
#     types = df.dtypes.apply(lambda x: 'string' if x == 'object' else 'numeric').tolist()
#
#     for i, t in enumerate(types):
#         col = df.iloc[:, i]
#
#         if t == "string" or t == "categorical":
#             # For string or categorical types, use CountVectorizer
#             vectorizer = CountVectorizer(min_df=1, token_pattern='\S+')
#             feature_list.append(vectorizer.fit_transform(col).toarray())
#             transform_list.append(vectorizer)
#         else:
#             # For numeric types, use FunctionTransformer with custom function tryParse
#             vectorizer = FunctionTransformer(tryParse)
#             feature_list.append(np.array(vectorizer.transform(col)).T)
#             transform_list.append(vectorizer)
#
#     processed_df = pd.DataFrame(np.hstack(feature_list))
#     return processed_df, transform_list


def manual_categorical_imputation(df, categorical_columns):
    df=df.reset_index(drop=True)
     # Step 1: Fill categorical missings with "missing"
    df[categorical_columns] = df[categorical_columns].fillna("missing")
    assertion = df.isin(['missing']).any().any()
    # Step 2: Use OneHotEncoder
    # encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    # encoded_data = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
    encoder = OneHotEncoder(handle_unknown='ignore')
    # fit and transform color column
    one_hot_array = encoder.fit_transform(df[categorical_columns]).toarray()

    # create new dataframe from numpy array
    encoded_data = pd.DataFrame(one_hot_array, columns=encoder.get_feature_names_out(), index=df.index)

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
            encoded_data.drop(columns=[missing_indicator_col], inplace=True)


    df.drop(columns=categorical_columns,inplace=True)
    df = pd.concat([df, encoded_data], axis=1)

    return df

def drop_label_with_null(df, column_name):
    # Drop rows where the specified column is null
    df_cleaned = df.dropna(subset=[column_name])

    return df_cleaned

def encoding(test_df):


    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        # handle_missing="ignore"
    )
    ohe.fit_transform(test_df)
    return test_df

def drop_categorical_columns(df,conversion=False,featurize=False):
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object','category']).columns.tolist()
    return_df = df

    if conversion==True:
        # this is to avoid droping int and float mixed type columns since they will be considered objects
        for col in categorical_columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        # after those are taken care of we can drop the columns that are still object
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            # Find the most common value in the column
            most_common_value = df[col].mode().iloc[0]

            # Map the non-null column values accordingly
            df[col] = df[col].apply(
                lambda x: 1 if pd.notna(x) and x == most_common_value else (0 if pd.notna(x) else x))

        return_df=df.copy()

    elif featurize==True:
        # this is to avoid droping int and float mixed type columns since they will be considered objects
        for col in categorical_columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        # after those are taken care of we can drop the columns that are still object
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            if df[col].nunique() > 20:
             df.drop(columns=[col],inplace=True)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

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


# def drop_categorical_columns(df,featurize=False):
#     # Identify categorical columns
#     categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#     # Create a TfidfVectorizer
#     tfidf_vectorizer = TfidfVectorizer()
#
#     if featurize==True:
#         # Iterate over each categorical column
#         for col in categorical_columns:
#             # Create a temporary DataFrame with the current column
#             temp_df = df[[col]].copy()
#
#             # Convert the column to string and fill NaN values
#             temp_df[col] = temp_df[col].astype(str).fillna('')
#
#             # Fit and transform the current column
#             tfidf_matrix = tfidf_vectorizer.fit_transform(temp_df[col])
#
#             # Create a DataFrame with TF-IDF features for the current column
#             tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#
#             # Concatenate the TF-IDF DataFrame with the original DataFrame
#             df = pd.concat([df, tfidf_df], axis=1)
#     # Drop categorical columns from the DataFrame
#     df_without_categorical =  df.drop(categorical_columns, axis=1)
#
#     return df_without_categorical

def split_features_labels(df, label_column, task='Regression',situation='mean'):
    # Check if the specified label column exists in the DataFrame
    if label_column not in df.columns:
        print(f"Label column '{label_column}' not found in the DataFrame.")
        return None, None
    df_drop_label= df
    # Features (X) are all columns except the specified label column
    X = df_drop_label.drop(label_column, axis=1)
    if task=='classification':
        # Label (y) is the specified column
        # Create a new binary column based on the specified midpoint
        if len(df_drop_label[label_column].unique())<2:
            print(f"Label column '{label_column}'has only one label")
        elif len(df_drop_label[label_column].unique())>2:
            if situation=='mode':
                midpoint = df_drop_label[label_column].mode().iloc[0]
                df_drop_label.loc[:, label_column] = df_drop_label[label_column].apply(lambda x: 1 if x == midpoint else 0)
            elif situation=='median':
                midpoint = df_drop_label[label_column].median()
                df_drop_label.loc[:, label_column] = df_drop_label[label_column].apply(lambda x: 1 if x > midpoint else 0)
            else:
                midpoint = df_drop_label[label_column].mean()
                df_drop_label.loc[:, label_column] = df_drop_label[label_column].apply(
                    lambda x: 1 if x > midpoint else 0)
        else:
            print(f"Label column '{label_column}'is PERFECT for Classification")
    else:
        pass
        #print(f"Label column '{label_column}'is used for Regression")

    y = df_drop_label[label_column]

    return X, y
def get_single_value_columns(df):
    # Identify columns with only one unique value
    single_value_cols = df.columns[df.nunique() == 1].tolist()

    return single_value_cols
def read_names_file(file_path):
    feature_names = []

    with open(file_path, 'r') as file:
        for line in file:
            # Assuming feature names are listed in lines starting with a capital letter
            if re.match(r'^[A-Z]', line):
                feature_name = line.split()[0]
                feature_names.append(feature_name)

    return feature_names
def get_Xy(data,label):
    X = data.drop(label,axis = 1)
    y = data[label]
    return X,y
def get_simple_imputer_model_classification(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    # Get all column names with nulls
    columns_with_nulls = X_train.columns[X_train.isnull().any()]

    # Simple imputation using mean strategy for each column
    meanimputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Simple imputation using mean strategy for each column
    modeimputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    for col in columns_with_nulls:
        if X_train[col].nunique() > 2:
            X_train[col] = meanimputer.fit_transform(X_train[[col]]).flatten()
        else:
            X_train[col] = modeimputer.fit_transform(X_train[[col]]).flatten()

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()

    clf = SGDClassifier(
        loss="hinge",
        alpha=0.0000000001,
        max_iter=10000,
        fit_intercept=True,
        warm_start=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

def get_simple_imputer_model_regression(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    # Get all column names with nulls
    columns_with_nulls = X_train.columns[X_train.isnull().any()]

    # Simple imputation using mean strategy for each column
    meanimputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Simple imputation using mean strategy for each column
    modeimputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    for col in columns_with_nulls:
        if X_train[col].nunique() > 2:
            X_train.loc[:, col] = meanimputer.fit_transform(X_train[col].values.reshape(-1, 1))[:, 0]
        else:
            X_train.loc[:, col] = modeimputer.fit_transform(X_train[col].values.reshape(-1, 1))[:, 0]

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()



    clf = LinearRegression(fit_intercept = False).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    # score = clf.score(X_test, y_test)

    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

def get_knn_imputer_model_regression(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    # Get all column names with nulls
    original_columns_with_nulls = X_train.columns[X_train.isnull().any()]

    # Simple imputation using mean strategy for each column
    imputer = KNNImputer(keep_empty_features=True)
    imputed_X=pd.DataFrame(imputer.fit_transform(X_train))
    imputed_X.columns=X_train.columns

    new_columns_with_nulls = imputed_X.columns[imputed_X.isnull().any()]
    # Assert that there are no more null values in X_train
    assert not imputed_X.isnull().any().any()



    clf = LinearRegression(fit_intercept = False).fit(imputed_X,y_train)
    y_pred = clf.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    # score = clf.score(X_test, y_test)

    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time
def get_knn_imputer_model_classification(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    # Get all column names with nulls
    columns_with_nulls = X_train.columns[X_train.isnull().any()]

    # Simple imputation using mean strategy for each column
    imputer = KNNImputer(missing_values=np.nan)
    imputed_X=imputer.fit_transform(X_train)

    # Assert that there are no more null values in X_train
    assert not imputed_X.isnull().any().any()

    clf = SGDClassifier(
        loss="hinge",
        alpha=0.0000000001,
        max_iter=10000,
        fit_intercept=True,
        warm_start=True,
        random_state=42,
    )
    clf.fit(imputed_X, y_train)
    score = clf.score(X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

def get_miwei_imputer_model_classification(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    method='miwae'
    # Simple imputation using mean strategy for each column
    plugin = Imputers().get(method)
    imputed_X = plugin.fit_transform(X_train)
    # Assert that there are no more null values in X_train
    assert not imputed_X.isnull().any().any()

    clf = SGDClassifier(
        loss="hinge",
        alpha=0.0000000001,
        max_iter=10000,
        fit_intercept=True,
        warm_start=True,
        random_state=42,
    )
    clf.fit(imputed_X, y_train)
    score = clf.score(X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

def get_miwei_imputer_model_regression(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    method='miwae'
    # Simple imputation using mean strategy for each column
    plugin = Imputers().get(method)
    imputed_X = plugin.fit_transform(X_train)
    # Assert that there are no more null values in X_train
    assert not imputed_X.isnull().any().any()

    clf = LinearRegression(fit_intercept=False).fit(imputed_X, y_train)
    y_pred = clf.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

def get_naive_imputer_model_classification(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    # Drop rows with null values in the copy of X_train
    start_time_s = time.time()
    X_train.dropna(inplace=True)

    # Now X_train is a copy and not a view, and modifications won't raise the warning

    # Align y_train with the modified X_train
    y_train = y_train.iloc[X_train.index]

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()

    clf = SGDClassifier(
        loss="hinge",
        alpha=0.0000000001,
        max_iter=10000,
        fit_intercept=True,
        warm_start=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time


def get_naive_imputer_model_regression(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    start_time_s = time.time()
    # Drop rows with null values in the copy of X_train
    X_train.dropna(inplace=True)

    # Now X_train is a copy and not a view, and modifications won't raise the warning

    # Align y_train with the modified X_train
    y_train = y_train.iloc[X_train.index]

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()



    clf = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    # score = clf.score(X_test, y_test)

    end_time_s = time.time()
    naive_time = end_time_s - start_time_s
    return score, naive_time

def get_midas_imputer_model_regression(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='ignore')
    # after those are taken care of we can drop the columns that are still object
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # Get all column names with nulls
    columns_with_nulls = X_train.columns[X_train.isnull().any()]

    start_time_s = time.time()
    imputer = md.Midas(layer_structure=[256, 256], vae_layer=False, seed=89, input_drop=0.75)
    imputer.build_model(X_train, softmax_columns=categorical_columns)
    imputer.train_model(training_epochs=1)
    imputations = imputer.generate_samples(m=1).output_list
    X_train = imputations[0]

    # Assert that there are no more null values in X_train
    assert not X_train.isnull().any().any()



    clf = LinearRegression(fit_intercept = False).fit(X_train,y_train)
    score = clf.score(X_test, y_test)

    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

def get_midas_imputer_model_classification(df_train, df_test, label):
    X_train, y_train=get_Xy(df_train,label)
    X_test, y_test=get_Xy(df_test,label)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='ignore')
    # after those are taken care of we can drop the columns that are still object
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # Get all column names with nulls
    columns_with_nulls = X_train.columns[X_train.isnull().any()]

    start_time_s = time.time()
    imputer = md.Midas(layer_structure=[256, 256], vae_layer=False, seed=89, input_drop=0.75)
    imputer.build_model(X_train, softmax_columns=categorical_columns)
    imputer.train_model(training_epochs=1)
    imputations = imputer.generate_samples(m=1).output_list
    X_train = imputations[0]

    clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    end_time_s = time.time()
    simple_time = end_time_s - start_time_s
    return score, simple_time

if __name__ == '__main__':
    df = pd.read_csv('data/MELBOURNE_HOUSE_PRICES_LESS.csv')
    processed_df, transform_list = featurize_categorical_data(df)
    print(processed_df.head())



############################# Additional Active Clean Functions ########################
# class CSVLoader:
# 	"""
# 	This class provides a wrapper to load csv files into the system
# 	"""
#
# 	def __init__(self, delimiter=None, quotechar=None):
# 		"""
# 		You can create a CSV loader with a specified delimiter
# 		or quote character. Or it will just try them all.
# 		"""
# 		self.DELIMITERS = [',', '\t', ':', '~', '|']
# 		self.QUOTECHAR = ['"', "'", '|', ':']
#
# 		if delimiter != None:
# 			self.DELIMITERS = [delimiter]
#
# 		if quotechar != None:
# 			self.QUOTECHAR = [quotechar]
#
#
# 	def __load(self, fname, delimiter, quotechar):
# 		"""
# 		Internal method to load a CSVfile given a delimiter and
# 		quote character
# 		"""
# 		with open(fname,'rb') as file:
# 			try:
# 				return [r for r in csv.reader(file,
# 							  delimiter=delimiter,
# 							  quotechar=quotechar)]
# 			except:
# 				return None
#
# 		return None
#
#
# 	def __score(self,parsed_file):
# 		"""
# 		This method assigns a score to all of the parsed files.
# 		We count the variance in the row length
# 		"""
# 		lstcount = [len([r for r in row if r.strip() != '']) for row in parsed_file]
# 		rowstd = np.std(lstcount)
#
# 		#catch degenerate case
# 		if len(parsed_file[0]) == 1:
# 			return float("inf")
# 		else:
# 			return rowstd
#
#
# 	def loadFile(self, fname):
# 		"""
# 		External method to load a file
# 		"""
# 		parsed_files = []
#
# 		#try out all of the options in the delimiter and quotechar set
# 		for delimiter in self.DELIMITERS:
# 			for quotechar in self.QUOTECHAR:
# 				loaded = self.__load(fname, delimiter, quotechar)
# 				if not loaded == None:
# 					parsed_files.append(((delimiter,quotechar), loaded))
#
# 		#score each of the parsed files
# 		scored_parses = [(self.__score(p[1]), p[0], p[1]) for p in parsed_files]
#
# 		scored_parses.sort()
#
# 		self.delim = scored_parses[0][1]
#
# 		#worst case just split on spaces
# 		if len(scored_parses[0][2][0]) == 1:
# 			return self.__load(fname, ' ', '"')
# 		else:
# 			return scored_parses[0][2]
#
#
#
#
