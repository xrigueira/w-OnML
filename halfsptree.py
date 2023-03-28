"""Half-space trees are an online variant of isolation forests. 
They work well when anomalies are spread out. However, they do
not work well if anomalies are packed together in windows."""

import data

from river import compose
from river import metrics
from river import anomaly
from river import preprocessing

dataset = data.labeled_901()

model = compose.Pipeline(
    compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees(seed=24)
)

metric = metrics.ROCAUC()

for x, y in dataset:
    score = model.score_one(x)
    model = model.learn_one(x)
    metric = metric.update(y, score)

print(metric)