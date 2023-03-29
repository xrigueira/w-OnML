import data
import numpy as np
import pandas as pd

from pprint import pprint
from tictoc import tictoc


if __name__ == '__main__':

    # imputation_knn(station=901)

    dataset = data.labeled_901()

    from river import compose
    from river import metrics
    from river import linear_model

    model = compose.Pipeline(
        compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
        linear_model.LogisticRegression()
    )

    metric = metrics.ROCAUC()

    for x, y in dataset:
        y_pred = model.predict_proba_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)
        # print(model.debug_one(x))

    print(metric)
