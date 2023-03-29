"""Working with imbalanced data"""

import data
import collections

from river import metrics
from river import compose
from river import evaluate
from river import linear_model
from river import preprocessing
from river import optim
from river import imblearn

dataset = data.labeled_901()

counts = collections.Counter(y for _, y in dataset)

for c, count in counts.items():
    print(f'{c}: {count} ({count / sum(counts.values()):.5%})')

model = compose.Pipeline(
    compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
    imblearn.RandomSampler(
        classifier=linear_model.LogisticRegression(),
        desired_dist={0: .8, 1: .2},    # Samples data to contain 80% of 0s and 20% of 1s
        sampling_rate=.01,              # Trains with 1% of the data
        seed=42
    )
)

metric = metrics.ROCAUC()

for x, y in dataset:
    y_pred = model.predict_proba_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)
    # print(model.debug_one(x))

print(metric)


