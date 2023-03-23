"""This file gets the percentage of gaps in each variable
of each merged file"""

import os
import time
import pandas as pd

def tictoc(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time() - t1
        print(f'{func.__name__} ran in {round(t2, ndigits=2)} seconds')
    return wrapper

@tictoc
def gapper():

    gaps = pd.DataFrame(columns=['Station', 'Variable', 'Perc_Gaps', 'Perc_Gaps_Cropped'])

    # Extract file names
    files = os.listdir('data')[:35]

    for f in files:

        station = f[7:10]
        df = pd.read_csv(f'data/{f}', sep=',', parse_dates=['date'], encoding='utf-8')

        # Get the columns
        variables = df.columns[9:]

        for v in variables:

            # Whole time series
            percent_missing = df[v].isnull().sum() * 100 / len(df)

            # Since a certain date
            df = df.loc[(df['date'] >= '2008-01-01 00:00:00')]
            percent_missing_croped = df[v].isnull().sum() * 100 / len(df)

            # Read again
            df = pd.read_csv(f'data/{f}', sep=',', parse_dates=['date'], encoding='utf-8')

            # Add the results of each iteration to the dataframe
            gaps.loc[len(gaps.index)] = [station, v, percent_missing, percent_missing_croped]

    # Save the results
    gaps.to_csv(f'data/gaps.csv', sep=',', encoding='utf-8', index=False)

@tictoc
def columner():

    cols = pd.DataFrame(columns=['Station', 'Rows_Missing1', 'Rows_Missing2', 'Rows_Missing3', 'Rows_Missing4', 'Rows_Missing5', 'Rows_Missing6', 'Rows_Missing7', 'Rows_Missing8'])

    # Extract file names
    files = os.listdir('data')[:35]
    
    for f in files:
        
        station = f[7:10]
        
        df = pd.read_csv(f'data/{f}', sep=',', parse_dates=['date'], encoding='utf-8')

        # Get the number of missing values per row in the database
        missing = df.isnull().sum(axis=1).tolist()

        # Count how many times a row is missing one, two, three, four or five variables
        one_var, two_var, three_var, four_var = missing.count(1) / len(df), missing.count(2) /len(df), missing.count(3) / len(df), missing.count(4) / len(df), 
        five_var, six_var, seven_var, eight_var = missing.count(5) / len(df), missing.count(6) / len(df), missing.count(7) / len(df), missing.count(8) / len(df)
        
        # Add the results of each iteration to the dataframe
        cols.loc[len(cols.index)] = [station, one_var, two_var, three_var, four_var, five_var, six_var, seven_var, eight_var]

    # Save the results
    cols.to_csv(f'data/cols.csv', sep=',', encoding='utf-8', index=False)


if __name__ == '__main__':

    columner()
