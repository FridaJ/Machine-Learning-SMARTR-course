import numpy as np
import pandas as pd

# To install packages, activate conda command by typing in terminal:
# > . /Users/xjacfr/opt/anaconda3/bin/activate && conda activate /Users/xjacfr/opt/anaconda3;
# Then use the conda install command given on the package web site.
# Lastly, install the package in Python (Preferences/Project/Python Interpreter/ <plus button>)

# txt files with data from course:
df = pd.read_csv("SmarterML_Training.Input", delim_whitespace=True, header=None).values
X = pd.DataFrame(df).values   # -> df, and using .values to avoid warnings in sklearn, our X feature matrix
labels = pd.read_csv("SmarterML_Training.Label", header=None).values
labels = pd.DataFrame(labels)
y = labels.iloc[:,0].values # A numpy array, target values

# Read evaluation set:
df_eval = pd.read_csv("SmarterML_Eval.Input", delim_whitespace=True, header=None).values

# 23/4
# Added the 5 x 5 cross-validation subsetting below (including lots of time trying to solve a problem with computer
# not installing packages... spent 7 h on trying to fix it, but incompatible packages... Found out later that I can
# use sklearn for CV splits. ONLY USE cross_val_predict!

# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5) # Define the split - into 5 folds
# from sklearn.model_selection import cross_val_score
# Perform 5-fold cross validation with for-loop and check scores with:
# scores = cross_val_score(<model>, X, y, cv=5)
# print("Cross-validated scores:", scores)

# For making cross validation later:
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Before bedtime 3 am Sunday morning, I want to write down my list of operations for each method tested:
# 0. Before any testing of methods, set aside 20% of the training set for validation purposes. Also create input X
# and y. Import cross_val_predict, cross_val_score and cross_validate.
# 1. Import estimator method
# 2. Use cross_val_predict with estimator and data/targets, cv=5, shuffle might be good
# 3. Calculate mean and std of scores from cross_val_predict and evaluate the parameters
# 4. When satisfied with parameters, use estimator.fit on the whole training set X
# 5. Evaluate results with cross_val_predict or score and also % misclassification
# 6. Check the first 50 entries of evaluation set and compare to other estimators.

# 19/4
# Started by importing PyTorch and Pandas and set up a new working environment with some problems. (update: Will not use PyTorch just yet) 
# Made the first code, just reading the input files with pandas and concatenating them together to one dataframe. Changed the name of the
# columns to "label" and the string integers "1" to "52".

# 22/4
# Decided for an outline, to try different algorithms for the classification problem.
# 1. Naive Bayes (linear)
# 2. KNN (non-linear)
# 3. Random Forest (ensemble)
# 4. Deep learning (neural network)
# Maybe combine the above with or without some kind of feature selection also.

print("\n", "-----Naive Bayes, test set 20%, 5 x 5 cv -----", "\n")

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

preds = cross_val_predict(gnb, X_train, y_train, cv=5)
print("Number of mislabeled points out of a total %d points: %d" % (X_train.shape[0], (preds != y_train).sum()))
print("Error percentage:", 100*(preds != y_train).sum()/X_train.shape[0])

scores = cross_val_score(gnb, X_train, y_train, cv=5)
m = np.mean(scores)
sd = np.std(scores)
print("The mean score after cv of training set is", m, "with std", sd, ".")

test_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points: %d" % (X_test.shape[0], (test_pred != y_test).sum()))
print("Error percentage:", 100*(test_pred != y_test).sum()/X_test.shape[0])

scores = cross_val_score(gnb, X_test, y_test, cv=5)
m = np.mean(scores)
sd = np.std(scores)
print("The mean score after cv of testing set is", m, "with std", sd, ".")

eval_pred = gnb.predict(df_eval) # Predicts 0 or 1 for the evaluation set
print("First 50 entries of eval_pred: \n", eval_pred[0:50])

# Results: Not very good with around 25% missclassification. Tried different test set sizes but same result.
# Look for a better method!

# 23/4
# Read a lot on scikit-learn page about different methods, mostly about KNN and logistic regression and their related
# functions with parameters. Maybe kernel method together with logistic regression. Thinking about doing some regularization of the data before using logistic regression or KNN.
# A function sparsity() converts the coeff matrix to a sparse matrix, seems interesting. Scikit-learn page suggests using
# liblinear solving method for logistic regression if dataset is small, so I will try doing that. There is a logistic
# regression function with built-in cross validation, I will try it below: (3 h) Liblinear didn't converge, I changed
# solver to sag instead. Then it didn't converge at all! Changed back to default and used built-in cv with increased max_iter. (23/4)

print("\n", "----- Logistic regression, 5 x 5 CV -----", "\n")

from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5, solver='liblinear', random_state=0, max_iter=200)

# preds = cross_val_predict(clf, X_train, y_train, cv=5)
preds = clf.fit(X_train, y_train).predict(X_train)
print("Number of mislabeled points out of a total %d points: %d" % (X_train.shape[0], (preds != y_train).sum()))
print("Error percentage:", 100*(preds != y_train).sum()/X_train.shape[0])

scores = cross_val_score(clf, X_train, y_train, cv=5)
m = np.mean(scores)
sd = np.std(scores)
print("The mean score after cv of training set is", m, "with std", sd, ".")

test_pred = clf.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points: %d" % (X_test.shape[0], (test_pred != y_test).sum()))
print("Error percentage:", 100*(test_pred != y_test).sum()/X_test.shape[0])

scores = cross_val_score(clf, X_test, y_test, cv=5)
m = np.mean(scores)
sd = np.std(scores)
print("The mean score after cv of testing set is", m, "with std", sd, ".")

eval_pred = clf.predict(df_eval) # Predicts 0 or 1 for the evaluation set
print("First 50 entries of eval_pred: \n", eval_pred[0:50])

# Searched for ways to save predictions in txt files from dataframes, and made a file from the logistic
# regression method.

eval_pred_temp = pd.DataFrame(data=eval_pred)
eval_pred_temp.to_csv('eval_pred1.txt', header=None, index=None, sep=' ', mode='a')

# The logistic regression with CV always gives 300 misclassifications, so I guess 300 of the samples are different
# in some way, maybe making the data non-parametric.
# Will now try KNN.

print("\n", "----- KNN, brute force method, n = 4, weights = distance -----", "\n")

from sklearn.neighbors import KNeighborsClassifier

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
neigh = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree', weights='uniform')

preds = cross_val_predict(neigh, X_train, y_train, cv=5)
print("Number of mislabeled points out of a total %d points: %d" % (X_train.shape[0], (preds != y_train).sum()))
print("Error percentage:", 100*(preds != y_train).sum()/X_train.shape[0])

scores = cross_val_score(neigh, X_train, y_train, cv=5)
m = np.mean(scores)
sd = np.std(scores)
print("The mean score after cv of training set is", m, "with std", sd, ".")

test_pred = neigh.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points: %d" % (X_test.shape[0], (test_pred != y_test).sum()))
print("Error percentage:", 100*(test_pred != y_test).sum()/X_test.shape[0])

scores = cross_val_score(neigh, X_test, y_test, cv=5)
m = np.mean(scores)
sd = np.std(scores)
print("The mean score after cv of testing set is", m, "with std", sd, ".")

eval_pred = neigh.predict(df_eval) # Predicts 0 or 1 for the evaluation set
print("First 50 entries of eval_pred: \n", eval_pred[0:50])

# Using weights = 'uniform' (default), KNN generated about 300 misclassifications of X, but changing this parameter
# to 'distance' (letting the closest neighbours contribute more) made the error rate 0%! Which was obviously wrong,
# which I saw after adding cross validation. Also tried the tree algorithms but no success (23/4).



#####----- Deposit 1 of the project -----#####

# Use the logistic regression prediction?
