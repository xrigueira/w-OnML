"""A pipeline is essentially a list of estimators that are applied in sequence. 
The only requirement is that the first n - 1 steps be transformers. 
The last step can be a regressor, a classifier, a clusterer, a transformer, etc"""

from river import compose, linear_model, preprocessing, feature_extraction

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    feature_extraction.PolynomialExtender(),
    linear_model.LinearRegression(),
)

""" in a pipeline, learn_one updates the supervised parts, whilst predict_one 
updates the unsupervised parts. It's important to be aware of this behavior"""

from river import datasets

dataset = datasets.TrumpApproval()
x, y = next(iter(dataset))

print(model['StandardScaler'].means)

print(model.predict_one(x)) # The prediction is null because each weight of the linear regressión is 0

print(model['StandardScaler'].means) # However, the means of each feature has been updated,  even though we called predict_one and not learn_one


