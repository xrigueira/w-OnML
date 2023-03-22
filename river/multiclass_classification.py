"""Classification is about predicting an outcome from a fixed list 
of classes. The prediction is a probability distribution that assigns 
a probability to each possible outcome"""

from pprint import pprint
from river import datasets

dataset = datasets.ImageSegments()
pprint(dataset)

# The dataset is a streaming dataset that can be looped over
for x, y in dataset:
    pass

x, y = next(iter(dataset)) # check out the first example
print(y)

"""A multiclass classifier's goal is to learn how to predict a class y
from a bunch of features x. We'll attempr to do this with a decision tree"""

from river import tree

model = tree.HoeffdingTreeClassifier()
print(model.predict_proba_one(x))

"""The reason why the output dictionary is empty is because the model hasn't
seen any data yet. It isn't aware of the dataset whatsoever. If this were a 
binary classifier, then it would output a probability of 50% for True and False 
because the classes are implicit. But in this case we're doing multiclass 
classification."""

model.learn_one(x, y)
print(model.predict_proba_one(x))

"""Path is the only class that it as seen. If we train it in an online matter
with more classes coming it, the results will adapt.
Typically, an online model makes a prediction, and then learns once the ground 
truth reveals itself. The prediction and the ground truth can be compared to 
measure the model's correctness. If you have a dataset available, you can loop over 
it, make a prediction, update the model, and compare the model's output with the 
ground truth. This is called progressive validation"""

from river import metrics

model = tree.HoeffdingTreeClassifier()
metric = metrics.ClassificationReport()

for x, y in dataset:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    if y_pred is not None:
        metric.update(y, y_pred)

print(metric)

# This evaluation method can also be implemented in one line with a dedicated function
from river import evaluate
print(evaluate.progressive_val_score(dataset, model, metric))


