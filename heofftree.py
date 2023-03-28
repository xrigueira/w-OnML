import data

from river import tree
from river import compose
from river import metrics
from river import evaluate

dataset = data.labeled_901()

model = compose.Pipeline(
    compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
    tree.HoeffdingTreeClassifier(grace_period=200)
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)

print(metric)