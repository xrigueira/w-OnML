import pandas as pd

""""Deletes the undesired variables in each database"""

station = 916
variables = ['absorbance']

# Read the dfs
merged = pd.read_csv(f'data/merged_{station}.csv', sep=';', encoding='utf-8')
normed = pd.read_csv(f'data/normed_{station}.csv', sep=';', encoding='utf-8')

for var in variables:

    merged.drop(f'{var}_{station}', inplace=True, axis=1)
    normed.drop(f'{var}_{station}', inplace=True, axis=1)

# Save the files
merged.to_csv(f'data/merged_{station}.csv', sep=';', encoding='utf-8', index=False)
normed.to_csv(f'data/normed_{station}.csv', sep=';', encoding='utf-8', index=False)