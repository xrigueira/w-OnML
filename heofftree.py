import data

from pprint import pprint
from river import compose
from river import tree

dataset = data.labeled_901()

model = compose.Pipeline(
    compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
    tree.HoeffdingTreeClassifier(grace_period=50)
)

for x, y in dataset:
    model.learn_one(x, y)
    # print(model.debug_one(x))