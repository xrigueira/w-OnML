import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# https://realpython.com/logistic-regression-python/#logistic-regression-in-python-with-scikit-learn-example-1

def custom_train_test_split(X, y, train_size):
    """This function split the data at the 'train size' length. The test set 
    would be all items which index is smaller then this number, while the test 
    size all those items with an index above.
    ----------
    Argument:
    X (np.array): predictor variables.
    y (np.array): labels or target variable.
    train_size (float): defines the size of the train and test sets.
    
    Return:
    X_train: (np.array): variables train set.
    X_test: (np.array): variables test set.
    y_train: (np.array): target train set.
    y_test: (np.array): target test set."""

    # Define train sets
    X_train = X[:int(train_size*len(X))]
    y_train = y[:int(train_size*len(y))]

    # Define test sets
    X_test = X[int(train_size*len(X)):]
    y_test = y[int(train_size*len(y)):]

    return X_train, X_test, y_train, y_test

# Read the data
data = pd.read_csv('data/labeled_901_cle.csv', sep=',', encoding='utf-8')

# Convert variable columns to np.ndarray
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Split the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
# X_train, X_test, y_train, y_test = custom_train_test_split(X, y, train_size=0.8)

#%% Implement grid search
# Define the parameters to iterate over
param_grid = {'penalty': ['l1', 'l2'], 'fit_intercept': [True, False], 'C': [0.1, 1, 10, 50, 100],
            'intercept_scaling': [0.1, 1, 5, 10], 'class_weight': ['balanced', None], 'max_iter': [64, 128, 256, 1024]}

# Call grid search
grid_search = GridSearchCV(LogisticRegression(solver='liblinear', random_state=0), param_grid=param_grid, cv=5)

# Fit grid search to the training data
grid_search.fit(X_train, y_train)

# Get best params
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# Make predictions on the testing data
y_hat = best_model.predict(X_test)

# Get the accuracy of the model
accuracy = accuracy_score(y_test, y_hat)
print('Accuracy', accuracy)

# Get the number of rows labeled as anomalies in y_test
print('Number of anomalies', len([i for i in y_test if i==1]))

# Display the confusion matrix
confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
print(confusion_matrix)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')
plt.show()


# # Evaluate with my own metric
# from main import Metric

# class Metric2(Metric):
#     def __init__(self, labels, predicted_labels, anomaly_tail) -> None:
#         self.labels = labels
#         self.predicted_labels = predicted_labels
#         self.anomaly_tail = anomaly_tail

# metric = Metric2(labels=y_test, predicted_labels=y_hat, anomaly_tail=0.25)
# result = metric.match_percentage()

# print(result)
