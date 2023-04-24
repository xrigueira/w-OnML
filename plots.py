"""This file gets the percentage of gaps in each variable
of each merged file"""

import os
import random
import numpy as np
import pandas as pd
import plotly.express as px

from tictoc import tictoc
from matplotlib import pyplot as plt
plt.style.use('ggplot')

@tictoc
def gap_percent_per_variable():
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
def gap_percent_per_rwo():
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

@tictoc
def gap_analyzer():
    """This function analyzes the number of gaps, their lenght and the number of variables affected in each case.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    
    gaps = pd.DataFrame()

    # Define file names
    files = ['901', '902', '904', '905', '906', '907', '910', '916']
    
    for f in files:
        
        # Read the database
        df = pd.read_csv(f'data/labeled_{f}.csv', sep=',', encoding='utf-8')

        # Get the number of missing values per row in the database
        missing = df.isnull().sum(axis=1).tolist()
        
        gaps[f] = missing
        
        # Get the length of each gap
        gap_lenghts = []
        gap_counter = 0
        for i in missing:
            if i != 0:
                gap_counter += 1
            
            elif i == 0 and gap_counter !=0:
                gap_lenghts.append(gap_counter)
                gap_counter = 0
        
        plt.plot(gap_lenghts, color=tuple(random.random() for _ in range(3)))
        plt.title(f'Gap length {f}')
        plt.ylabel('Length of each gap')
        plt.show()
    
    # Plots
    plt.hist(gaps.values, alpha=0.80, label=gaps.columns)
    plt.title('Missing values boxplot')
    plt.ylabel('Missing values per row')
    plt.legend()
    plt.show()

    # Also get the number of anomalies that have missing values and the percentage with respect to the total number of anomalies: To program this: count number of rows that have gaps and have a label == 1

@tictoc
def label_analyzer():
    """This function studies the number of rows labeled as anomalies and those which have missing values but
    are also labeled as anomalous.
    ----------
    Arguments:
    None
    
    Return:
    None"""
    # Define file names
    files = ['901', '902', '904', '905', '906', '907', '910', '916']
    
    # Initialize lists to store the counts
    anomalies_counts = []
    missing_anomalies_counts = []
    for f in files:
        
        # Read the database
        df = pd.read_csv(f'data/labeled_{f}.csv', sep=',', encoding='utf-8')
        
        # Count the number of anomalies and missing anomalies
        anomalies = (df['label'] == 1).sum()
        missing_anomalies = ((df['label'] == 1) & (df.isna().any(axis=1))).sum()
        
        # Append the counts to the lists
        anomalies_counts.append(anomalies)
        missing_anomalies_counts.append(missing_anomalies)
    
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Define the x positions of the bars
    x_positions = range(len(files))

    # Define the width of the bars
    bar_width = 0.4
    
    # Create the bars
    anomalies_bars = ax.bar(x_positions, anomalies_counts, bar_width, label='Anomalies')
    missing_anomalies_bars = ax.bar([x + bar_width for x in x_positions], missing_anomalies_counts, bar_width, label='Missing Anomalies')

    # Set the x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(files)

    # Add a legend
    ax.legend()

    # Add axis labels and a title
    ax.set_xlabel('File')
    ax.set_ylabel('Count')
    ax.set_title('Anomalies and Missing Anomalies by File')

    # Show the plot
    plt.show()

def multivar_plotter():
    pass

if __name__ == '__main__':

    multivar_plotter()
