import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
import sys

# Save the current stdout for later use
original_stdout = sys.stdout

# Open a file in write mode to save the print statements
with open('output.txt', 'w') as file:
    # Redirect stdout to the file
    sys.stdout = file

    running_time = 0
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('CommunitiesAndCrime.csv')

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
    ACM_model = np.zeros(d)

    # Fill missing values in X_train with the maximum value of each column
    X_train_repair = X_train.fillna({col: max_min_values[col]['max'] for col in X_train.columns[X_train.isnull().any()]})

    def loss_function(w, X, y):
        # Compute the mean squared error (MSE) loss
        mse = np.mean((np.dot(X, w) - y) ** 2)
        return mse

    def gradient_function(w, X, y):
        # Compute the gradient of the loss function with respect to the weights
        X_train_repaired_end1 = X_train.copy()
        X_train_repaired_end2 = X_train.copy()
        for i in range(X_train.shape[0]):  # Iterate through each row
            for j in range(X_train.shape[1]):  # Iterate through each column
                if pd.isnull(X_train.iloc[i, j]):  # Check if the value is missing
                    if w[j] > 0:  # If the corresponding weight is positive
                        X_train_repaired_end1.iloc[i, j] = max_min_values[X_train.columns[j]]['max']  # Impute with max
                        X_train_repaired_end2.iloc[i, j] = max_min_values[X_train.columns[j]]['min']
                    else:
                        X_train_repaired_end1.iloc[i, j] = max_min_values[X_train.columns[j]]['min']  # Impute with min
                        X_train_repaired_end2.iloc[i, j] = max_min_values[X_train.columns[j]]['max']

        # Calculate w^Tx_i - y_train_i for each row in both versions
        diff_end1 = abs(X_train_repaired_end1.dot(w) - y_train)
        diff_end2 = abs(X_train_repaired_end2.dot(w) - y_train)

        # Select the rows where the difference is larger in X_train_supremum
        X_train_supremum = X_train_repaired_end1.copy()
        X_train_supremum[diff_end2 > diff_end1] = X_train_repaired_end2[diff_end2 > diff_end1]
        gradient = -2/n * X_train_supremum.T.dot(y_train - X_train_supremum.dot(w))

        return gradient

    def callback_function(w):
        global iteration
        global gradient_norm_prev
        global X_train_repair
        global y_train

        gradient = gradient_function(w, X_train_repair.values, y_train.values)
        gradient_norm = np.linalg.norm(gradient)

        iteration += 1
        gradient_norm_prev = gradient_norm

    def main():
        # Example parameters
        learning_rate = 0.0001
        max_iterations = 1000
        tolerance = 1e-5

        global iteration
        global gradient_norm_prev
        global X_train_repair
        global y_train

        # Randomly initialize the model parameters
        initial_w = np.zeros(d)

        iteration = 1
        gradient_norm_prev = float('inf')
        start_time = time.time()
        # Perform gradient descent using minimize
        result = minimize(loss_function, initial_w, jac=gradient_function, args=(X_train_repair.values, y_train.values),
                          method='BFGS', options={'maxiter': max_iterations, 'gtol': tolerance}, callback=callback_function)
        # Get the learned model parameters
        end_time = time.time()
        running_time = end_time - start_time
        learned_w = result.x
        training_loss = result.fun  # Loss value at the minimum

        print("\nLearned Model Parameters:", learned_w)
        print("Training Loss:", training_loss)
        print(running_time)

    if __name__ == "__main__":
        main()

  


