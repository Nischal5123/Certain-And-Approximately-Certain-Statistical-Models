import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import time
import sys
import warnings

# Filter out FutureWarnings and UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('CommunitiesAndCrime.csv')
running_time = 0
# Create a dictionary to store the max and min of each incomplete column
max_min_values = {}

# Iterate through each column with missing values
for column in df.columns[df.isnull().any()]:
    # Calculate the max and min of the non-missing values in the column
    max_value = df[column].max(skipna=True)
    min_value = df[column].min(skipna=True)
    # Store the max and min values in the dictionary
    max_min_values[column] = {'max': max_value, 'min': min_value}

# Separate the features (X) and the label (y)
X = df.drop(columns=['ViolentCrimesPerPop'])
y = df['ViolentCrimesPerPop']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n, d = X_train.shape  # Get the number of samples and features from the feature matrix

# Number of versions to create
m = 100  # Replace with your desired number of versions

    
# Your existing code for creating X_train_repaired_versions and losses goes here...
X_train_repaired_versions = []
losses = []
start_time = time.time()
for i in range(m):
    # Create a new version of X_train by copying the original
    X_train_repaired = X_train.copy()
        
    # Iterate through each column with missing values
    for column in X_train.columns[X_train.isnull().any()]:
        # Get the maximum and minimum values for the column
        max_value = max_min_values[column]['max']
        min_value = max_min_values[column]['min']
        
        # Iterate through each row in the column
        for index, value in X_train[column].items():
            # Check if the value is missing
            if pd.isnull(value):
                # Impute the missing value with a random value between the maximum and minimum
                imputation_value = np.random.choice([max_value, min_value])
                X_train_repaired.at[index, column] = imputation_value
    X_train_repaired_np = X_train_repaired.to_numpy()
        # Store the new version of X_train
    X_train_repaired_versions.append(X_train_repaired_np)
        
    # Create and fit the linear regression model for the current version
    model = LinearRegression().fit(X_train_repaired_np, y_train)
    # Predict the target values for the current version
    y_pred = model.predict(X_train_repaired_np)
    # Calculate the mean squared error (MSE) as the loss for the current version
    loss = np.sum(np.abs(y_train - y_pred) ** 2)
        
    # Store the loss for the current version
    losses.append(loss)

w = cp.Variable(d)

losses_cp = [
    cp.sum_squares(X_train_repaired_np @ w - y_train.to_numpy()) - min_loss
    for X_train_repaired_np, min_loss in zip(X_train_repaired_versions, losses)
]

# Calculate the maximum loss
loss_max = cp.maximum(*losses_cp)
# Define the optimization problem
prob = cp.Problem(cp.Minimize(loss_max))
prob.solve()
end_time = time.time()
running_time = end_time - start_time

# Drop rows with missing values from the test set
X_test_clean = X_test.dropna()
y_test_clean = y_test[X_test_clean.index]

# Evaluate the model w.value on X_test_clean and y_test_clean
y_pred_w = X_test_clean @ w.value
score_w_value = 1/n * np.sum(np.abs(y_test_clean - y_pred_w))

