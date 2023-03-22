import glob
import pandas as pd

"""This function merges/joines the separate variables in a single multivariate
file"""

def joiner(station):
    
    # List all csv files in /data
    csv_files = glob.glob(f'data/*{station}_col.csv')
    
    # Initialize an empty dataframe to store the merged data
    merged_df = pd.read_csv(csv_files[0], sep=';')
    first_var = str(merged_df.columns[1])
    
    # Loop through each file and append to the merged dataframe
    for i in csv_files[1:]:

        # Read the CSV file and store it in a temporary dataframe
        df = pd.read_csv(i, sep=';')
        
        cols = list(df.columns)
        cols.pop(1)
        merged_df = pd.merge(merged_df, df, on=cols)

    # Save the merged dataframe to a new CSV file
    merged_df.insert(9, first_var, merged_df.pop(first_var))
    merged_df.to_csv(f'data/merged_{station}.csv', sep=';', encoding='utf-8', index=False)
