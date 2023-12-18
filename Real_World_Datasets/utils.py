import pandas as pd
import missingno as msno
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
import scipy.sparse
import numpy as np
import re

class CSVLoader:
	"""
	This class provides a wrapper to load csv files into the system
	"""

	def __init__(self, delimiter=None, quotechar=None):
		"""
		You can create a CSV loader with a specified delimiter
		or quote character. Or it will just try them all.
		"""
		self.DELIMITERS = [',', '\t', ':', '~', '|']
		self.QUOTECHAR = ['"', "'", '|', ':']

		if delimiter != None:
			self.DELIMITERS = [delimiter]

		if quotechar != None:
			self.QUOTECHAR = [quotechar]


	def __load(self, fname, delimiter, quotechar):
		"""
		Internal method to load a CSVfile given a delimiter and
		quote character
		"""
		with open(fname,'rb') as file:
			try:
				return [r for r in csv.reader(file,
							  delimiter=delimiter,
							  quotechar=quotechar)]
			except:
				return None

		return None


	def __score(self,parsed_file):
		"""
		This method assigns a score to all of the parsed files.
		We count the variance in the row length
		"""
		lstcount = [len([r for r in row if r.strip() != '']) for row in parsed_file]
		rowstd = np.std(lstcount)

		#catch degenerate case
		if len(parsed_file[0]) == 1:
			return float("inf")
		else:
			return rowstd


	def loadFile(self, fname):
		"""
		External method to load a file
		"""
		parsed_files = []

		#try out all of the options in the delimiter and quotechar set
		for delimiter in self.DELIMITERS:
			for quotechar in self.QUOTECHAR:
				loaded = self.__load(fname, delimiter, quotechar)
				if not loaded == None:
					parsed_files.append(((delimiter,quotechar), loaded))

		#score each of the parsed files
		scored_parses = [(self.__score(p[1]), p[0], p[1]) for p in parsed_files]

		scored_parses.sort()

		self.delim = scored_parses[0][1]

		#worst case just split on spaces
		if len(scored_parses[0][2][0]) == 1:
			return self.__load(fname, ' ', '"')
		else:
			return scored_parses[0][2]

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

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

def drop_label_with_null(df, column_name):
    # Drop rows where the specified column is null
    df_cleaned = df.dropna(subset=[column_name])

    return df_cleaned


def featurize_categorical_data(df):
    # Copy the input DataFrame to avoid modifying the original
    df_numeric = df.copy()

    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    for column in categorical_columns:
        # Identify null values in the current column
        null_mask = df[column].isnull()

        # Fit and transform the CountVectorizer on non-null values of the categorical column
        non_null_values = df[column][~null_mask]
        vectorized_data = vectorizer.fit_transform(non_null_values)

        # Convert the result to a DataFrame
        vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

        # Reset the index of the original DataFrame before concatenating
        df_numeric = df_numeric.reset_index(drop=True)

        # Join the DataFrames on the index
        df_numeric = pd.concat([df_numeric, vectorized_df], axis=1)

        # Replace null values in the original DataFrame
        df_numeric.loc[null_mask, column] = df[column][null_mask]

        # Drop the original categorical column
        df_numeric.drop(column, axis=1, inplace=True)

    return df_numeric

def drop_categorical_columns(df):


    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # for col in categorical_cols:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')

    # # Drop categorical columns from the DataFrame
    df_without_categorical = df.drop(categorical_cols, axis=1)

    return df_without_categorical
def split_features_labels(df, label_column, task='Regression'):
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
            midpoint = df_drop_label[label_column].mean()
            df_drop_label.loc[:, label_column] = df_drop_label[label_column].apply(lambda x: 1 if x > midpoint else 0)
        else:
            print(f"Label column '{label_column}'is PERFECT for Classification")
    else:
        print(f"Label column '{label_column}'is used for Regression")

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
if __name__ == '__main__':
    df = pd.read_csv('data/MELBOURNE_HOUSE_PRICES_LESS.csv')
    processed_df, transform_list = featurize_categorical_data(df)
    print(processed_df.head())
