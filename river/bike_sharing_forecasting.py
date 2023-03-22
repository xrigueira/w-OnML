from river import datasets
from pprint import pprint

dataset = datasets.Bikes()

for x, y in dataset:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break

# Start by simple linear regression on numeric features
from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import optim

# model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
# model |= preprocessing.StandardScaler()
# model |= linear_model.LinearRegression(optimizer=optim.SGD(lr=0.001))

# metric = metrics.MAE()

# evaluate.progressive_val_score(dataset, model, metric, print_every=20_000)

# Generally, a good idea for this kind of problem is to look at an average of the previous values
from river import feature_extraction
from river import stats

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric, print_every=20_000)

# Debugging
import itertools

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

# Simulating production conditions: predict, recieve ground truth, update
# This can be done using the moment and delay parameters in the progressive_val_score method
# The moment parameter determines which variable should be used as a timestamp, while the delay parameter controls the duration to wait before revealing the true values to the model.

import datetime as dt

evaluate.progressive_val_score(
    dataset=dataset,
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=20_000
)
