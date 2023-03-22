"""Train a logistic regression model to classify the
website phishing dataset"""

from pprint import pprint
from river import datasets

# Load the dataset
dataset = datasets.Phishing()
print(dataset)

# Sneak peek of the dataset
for x, y in dataset:
    pprint(x) # This contains the data
    print(y) # This contains the target, which is a boolean representing phishing or not phishing
    break

# Run the model on the dataset in a streaming fashion
from river import compose
from river import linear_model
from river import metrics
from river import preprocessing

model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())

metric = metrics.Accuracy()
counter = 0
for x, y, in dataset:
    y_pred = model.predict_one(x) # make a prediction
    metric = metric.update(y, y_pred) # update the metric
    model = model.learn_one(x, y) # make the model learn

print(metric)