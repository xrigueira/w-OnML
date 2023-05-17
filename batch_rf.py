import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Read the data
data = pd.read_csv('data/labeled_901_cle.csv', sep=',', encoding='utf-8')

# Convert variable columns to np.ndarray
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Split the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

random_search_option = True

if random_search_option == True:
    
    # Define the parameters to iterate over
    param_dist = {'n_estimators': [50, 75, 100, 125, 150, 175], 'max_depth': [1, 2, 3, 4, 5, 10, 15, 20, 50, None],
                'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}
    
    rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions = param_dist, n_iter=5, cv=5)
    
    rand_search.fit(X_train, y_train)
    
    # Get best params
    best_params = rand_search.best_params_
    best_model = rand_search.best_estimator_
    print('Best params', best_params, '| Best model', best_model)
    
    # Make predictions on the testing data
    y_hat = best_model.predict(X_test)
    

elif random_search_option == False:
    
    # Call the model
    model = RandomForestClassifier(random_state=0)

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
if random_search_option == True:
    confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
elif random_search_option == False:
    confusion_matrix = confusion_matrix(y_test, model.predict(X_test))

print(confusion_matrix)

