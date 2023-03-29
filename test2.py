
import data
import pandas as pd

from river import drift

"""Drift detection with ADaptative WINdowing (ADWIN)"""

# Read the database
df = pd.read_csv(f'data/labeled_901_cle.csv', sep=',', encoding='utf-8')

adwin = drift.ADWIN(delta=0.002)
drift = []

for i, val in enumerate(df.conductivity_901):
    in_drift, in_warning = adwin.update(val)
    if in_drift:
        # print(f'Drift detected at index {i}, input value: {val}')
        drift.append(i)

print(len(drift))