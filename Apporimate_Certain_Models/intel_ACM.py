import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import cvxpy as cp
import time
import sys
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('intel-sensor-data.csv')
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
X = df.drop(columns=['2'])
y = df['2']
y.loc[y == 0] = -1
y = y.reset_index(drop=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n, d = X_train.shape  # Get the number of samples and features from the feature matrix
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Number of versions to create
m = 100  # Replace with your desired number of versions

# Create m complete versions of X_train by imputing missing values randomly
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
                #imputation_value = np.random.choice([1e6, -1e6])
                X_train_repaired.at[index, column] = imputation_value
    X_train_repaired_np = X_train_repaired.to_numpy()
    # Store the new version of X_train
    X_train_repaired_versions.append(X_train_repaired_np)

    # Create an instance of SGDClassifier with hinge loss
    model = SGDClassifier(loss='hinge', alpha=1.0, max_iter=1000, tol=1e-3)

    # Fit the model to the training data
    # Fit the model to the training data
    model.fit(X_train_repaired_np, y_train)

    # Predict the target values for the current version
    y_pred = model.decision_function(X_train_repaired_np)

    # Calculate the hinge loss for the current version
    hinge_loss_value = np.sum(np.maximum(0, 1 - y_train * y_pred))
    # Calculate the L2 norm of the weight vector
    w_norm = np.linalg.norm(model.coef_)

    # Calculate the combined SVM loss function
    alpha = 1  # You can adjust the value of alpha
    loss = hinge_loss_value + alpha * (w_norm ** 2)
    
    # Store the loss for the current version
    losses.append(loss)

print(losses)
w = cp.Variable(d)
# Define the parameters
alpha = 1  # Regularization parameter


losses_cp = [
    cp.sum(cp.pos(1 - cp.multiply(y_train.to_numpy(), X_train_repaired_np @ w))) + 0.5 * alpha * cp.sum_squares(w) - min_loss
    for X_train_repaired_np, min_loss in zip(X_train_repaired_versions, losses)
]


# Calculate the maximum loss
loss_max = cp.maximum(*losses_cp)

# Define the optimization problem
prob = cp.Problem(cp.Minimize(loss_max))
prob.solve(solver=cp.SCS, verbose=True)
end_time = time.time()
running_time = end_time - start_time

# Drop rows with missing values from the test set
X_test_clean = X_test.dropna()
y_test_clean = y_test[X_test_clean.index]


# Predict the target values for the test set using the optimized w
y_pred_w = np.sign(X_test_clean @ w.value)  # Assuming binary classification with positive class as 1



