from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing

"""A batch learning example"""
# # Load the data
# dataset = datasets.load_breast_cancer()
# X, y = dataset.data, dataset.target

# # Define the steps of the model
# model = pipeline.Pipeline([
#     ('scale', preprocessing.StandardScaler()),
#     ('line_reg', linear_model.LogisticRegression(solver='lbfgs'))
# ])

# # Define a deterministic cross-validation procedure
# cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# # Compute the MSE values
# scorer = metrics.make_scorer(metrics.roc_auc_score)
# scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# # Display the average score and its standard deviation
# print(f'ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')

"""Online approach"""
from river import stream

# Compute the running mean and variance of the variable 'mean area'
n, mean , sum_of_squares, variance = 0, 0, 0, 0

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    n += 1
    old_mean = mean
    mean += (xi['mean area']) / n
    sum_of_squares += (xi['mean area'] - old_mean) * (xi['mean area'] - mean)
    variance = sum_of_squares / n
    
    print(f'Running mean: {mean:.3f}')
    print(f'Running variance: {variance:.3f}')

# Now we can scale the feature as the come along
from river import preprocessing

scaler = preprocessing.StandardScaler()

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer()):
    scaler = scaler.learn_one(xi)

# Implement linear regression with stochastic gradient descent
from river import linear_model
from river import optim

scaler = preprocessing.StandardScaler()
optimizer = optim.SGD(lr=0.01)
log_reg = linear_model.LogisticRegression(optimizer)

y_true = []
y_pred = []

for xi, yi in stream.iter_sklearn_dataset(datasets.load_breast_cancer(), shuffle=True, seed=42):

    # Scale the features
    xi_scaled = scaler.lean_one(xi).transform_one(xi)
    
    # Test the current model on the new "unobserved" sample
    yi_pred = log_reg.predict_proba_one(xi_scaled)
    # Train thee model with the new sample
    log_reg.learn_one(xi_scaled, yi)
    
    # Store the truth and the prediction
    y_true.append(yi)
    y_pred.append(yi_pred[True])

print(f'ROC AUC: {metrics.roc_auc_score(y_true, y_pred):.3f}')