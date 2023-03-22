"""Regression is about predicting a numeric output for a given sample. 
A labeled regression sample is made up of a bunch of features and a number. 
The number is usually continuous, but it may also be discrete. 
We'll use the Trump approval rating dataset as an example."""

from river import datasets

dataset = datasets.TrumpApproval()

# Check out the first sample
x, y = next(iter(dataset))
print(x)

"""A regression model's goal is to learn to predict a numeric target y from 
a bunch of features x. We'll attempt to do this with a nearest neighbors model"""

from river import neighbors

model = neighbors.KNNRegressor()
print(model.predict_one(x)) # It hasn't seen any data

# Train it on one sample
model = model.learn_one(x, y)
print(model.predict_one(x))

"""Typically, an online model makes a prediction, and then learns once the ground 
truth reveals itself. The prediction and the ground truth can be compared to 
measure the model's correctness. If you have a dataset available, you can loop over 
it, make a prediction, update the model, and compare the model's output with the 
ground truth. This is called progressive validation"""

from river import metrics

model = neighbors.KNNRegressor()

metric = metrics.MAE()

for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)

print(metric)

# This evaluation method can also be implemented in one line with a dedicated function
from river import evaluate

print(evaluate.progressive_val_score(dataset, model, metric))
