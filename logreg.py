import data
import pandas as pd

from pprint import pprint
from tictoc import tictoc

# Data imputation
def imputation_del(station):
    """Performs data "imputation" by deleting all those rows with missing values
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""

    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')

    # Remove all rows with missing values
    df = df.dropna()

    # Save the new dataframe
    df.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)

def imputation_iter(station):
    """Performs data imputation by iterating on all those rows with missing values
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')

    # See mfilterer.py
    # Interpolate
    df = (df.interpolate(method='polynomial', order=1)).round(2)
    
    # Save the new dataframe
    df.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)

def imputation_knn(station):
    pass

def imputation_trees(station):
    pass

if __name__ == '__main__':

    imputation_iter(station=901)

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
