"""This file gets the percentage of gaps in each variable
of each merged file"""

import os
import time
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.style.use('ggplot')

def tictoc(func):
    def wrapper(**kargs):
        t1 = time.time()
        func(**kargs)
        t2 = time.time() - t1
        print(f'{func.__name__} ran in {round(t2, ndigits=2)} seconds')
    return wrapper

@tictoc
def gapper():

    """This function gets the percentage of gaps in every variable of each dataset
    and saves it to a csv file.
    ----------
    Arguments:
    None

    Return:
    None"""

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

    """This function gets the percentage of rows mising a certain number of values [1,..,8]
    and saves it to a csv file.
    ----------
    Arguments:
    None
    
    Return:
    None"""

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


def numberBins(data):
    n = len(data) # number of observations
    range = max(data) - min(data) 
    numIntervals = np.sqrt(n) 
    width = range/numIntervals # width of the intervals
    
    return np.arange(min(data), max(data), width).tolist()

def histogram(data, station):
    plt.hist(data, bins=50)
    plt.title(f'Histogram {station}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    plt.show()


@tictoc
def analyzer(station):
    """This function analyzed the number of gaps, their lenght and the number of variables affected in each case.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')

    # Get the number of missing values per row in the database
    missing = df.isnull().sum(axis=1).tolist()
    
    histogram(data=missing, station=station)

    # Get the length of each gap
    gap_lenghts = []
    gap_counter = 0
    for i in missing:
        if i != 0:
            gap_counter += 1
        
        elif i == 0 and gap_counter !=0:
            gap_lenghts.append(gap_counter)
            gap_counter = 0

    # Summarize this results in a table. Let's say: what is the percentage of gaps that are just one row. To do this, program: number of 1s in gap_lengths/len(gap_lenghts)
    # I think the max gap that i can fill up would be 10 hours or less
    plt.plot(gap_lenghts)
    plt.show()

    # Also get the number of anomalies that have missing values and the percentage with respect to the total number of anomalies: To program this: count number of rows that have gaps and have a label == 1


if __name__ == '__main__':

    analyzer(station=904)
