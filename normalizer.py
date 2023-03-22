import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def normalizer(station):

    df = pd.read_csv(f'data/merged_{station}.csv', sep=';')

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the scaler to each column and transform them
    for i in (df.columns[9:]):
        df[f'{i}'] = scaler.fit_transform(df[[f'{i}']])

    df.to_csv(f'data/normed_{station}.csv', sep=';', encoding='utf-8', index=False)