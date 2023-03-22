"""Classification is about predicting an outcome from a fixed list of classes. 
The prediction is a probability distribution that assigns a probability to each 
possible outcome.

A labeled classification sample is made up of a bunch of features and a class. 
The class is a boolean in the case of binary classification. We'll use the phishing 
dataset as an example."""

from river import datasets

dataset = datasets.Phishing()

# Check out the first item in the dataset
x, y = next(iter(dataset))
print(x, y)

# Apply logistic regression
from river import linear_model

model = linear_model.LogisticRegression()
print(model.predict_proba_one(x)) # The model has not been trained so it outputs a 50-50 prob for each class

# Train the model
model = model.learn_one(x, y)
print(model.predict_proba_one(x)) 
print(model.predict_one(x)) # To get the most probable class

"""Typically, an online model makes a prediction, and then learns once the ground 
truth reveals itself. The prediction and the ground truth can be compared to measure 
the model's correctness. If you have a dataset available, you can loop over it, make a 
prediction, update the model, and compare the model's output with the ground truth. 
This is called progressive validation."""

from river import metrics

model = linear_model.LogisticRegression() # Load the model
metric = metrics.ROCAUC() # Load the metric

for x, y in dataset:
    y_pred = model.predict_proba_one(x) # Make one prediction
    model.learn_one(x, y) # Learn from the data
    metric.update(y, y_pred) # Update the model based on the result of the prediction and the ground truth
print(metric) # Evaluates the model

# Do the evaluation but with a dedicated function
from river import evaluate

model = linear_model.LogisticRegression()
metric = metrics.ROCAUC()

print(evaluate.progressive_val_score(dataset, model, metric))

"""A common way to improve the performance of a logistic regression is to scale the data. 
This can be done by using a preprocessing.StandardScaler"""

# Define a pipeline to organize the model
from river import compose
from river import preprocessing

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

metric = metrics.ROCAUC()
print(evaluate.progressive_val_score(dataset, model, metric))