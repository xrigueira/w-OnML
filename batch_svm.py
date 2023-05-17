import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('data/labeled_901_cle.csv', sep=',', encoding='utf-8')

# Convert variable columns to np.ndarray
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Split the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

# Call the model
model = svm.SVC(kernel='linear')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_hat = model.predict(X_test)

# Get the accuracy of the model
accuracy = accuracy_score(y_test, y_hat)
print('Accuracy', accuracy)

# Get the number of rows labeled as anomalies in y_test
print('Number of anomalies', len([i for i in y_test if i==1]))

# Display the confusion matrix
confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
print(confusion_matrix)
