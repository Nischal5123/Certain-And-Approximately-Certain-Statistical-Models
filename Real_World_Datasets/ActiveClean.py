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
    else:
        clf = SGDRegressor(penalty=None,max_iter=200)
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
                ypred = clf.predict(X_test)
                return total_cleaning, 0 if mean_squared_error(y_test, ypred) is None else mean_squared_error(y_test, ypred)

    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    if task == 'classification':
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

    else:
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if mean_squared_error(y_test, ypred) is None else mean_squared_error(y_test, ypred)


if __name__ == '__main__':
    print('Not implemented')