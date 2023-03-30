import pandas as pd

# Read the database
df = pd.read_csv(f'data/labeled_901.csv', sep=',', encoding='utf-8')

# Split the dataframe into two parts: one with missing values, and another without them
df_missing = df[df.isnull().any(axis=1)]
df_not_missing = df[~df.isnull().any(axis=1)]

# Drop the nonvariable columns
drop_columns = ['date', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'weekOrder', 'label']

df_missing = df_missing.drop(drop_columns, axis=1)
df_not_missing = df_not_missing.drop(drop_columns, axis=1)
print(df_missing)
# print(df_not_missing)

# Add the missing rows back to the original dataframe
df_imputed = pd.concat([df_not_missing, df_missing])
print(df_imputed)

# Add the dropped columns
df_dropped = df[drop_columns]
# print(df_dropped)
df_imputed = pd.concat([df_dropped, df_imputed], axis=1)
