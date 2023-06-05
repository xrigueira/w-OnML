"""This file gets the percentage of gaps in each variable
of each merged file"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from tictoc import tictoc
from matplotlib import ticker
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import seaborn as sb

def labeler(varname):

    """This function is just to label the plots correctly for the research papers."""

    if varname == 'ammonium':
        label_title = r'$NH_4$'
        label_y_axis = r'$NH_4$ ' + r'$(m*g/L)$'
    elif varname == 'conductivity':
        label_title = r'Conductivity'
        label_y_axis = r'Conductivity ' r'$(\mu*S/cm)$'
    elif varname == 'nitrates':
        label_title = r'$NO_{3^-}$'
        label_y_axis = r'$NO_{3^-}$ ' +r'$(m*g/L)$'
    elif varname == 'dissolved_oxygen':
        label_title = r'$O_2$'
        label_y_axis = r'$O_2$ ' r'$(m*g/L)$'
    elif varname == 'pH':
        label_title = r'pH'
        label_y_axis = r'pH'
    elif varname == 'temperature':
        label_title = r'Temperature'
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'water_temperature':
        label_title = r'River water remperature '
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'water_flow':
        label_title = r'Flow'
        label_y_axis = r'Flow ' + r'($m^3/s$)'
    elif varname == "turbidity":
        label_title = r'Turbidity'
        label_y_axis = r'Turbidity ' + r'(NTU)'
    elif varname == "rainfall":
        label_title = r'Rainfall'
        label_y_axis = r'Rainfall ' + r'(mm)'
    elif varname == "water_level":
        label_title = r'Water level'
        label_y_axis = r'Water level ' + r'(m)'
    
    return label_title, label_y_axis

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
def gap_percent_per_row():
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
def bubble_histogram():
    
    # Read the data
    data = np.load('gap_lengths.npy')
    
    # My data does not look with this idea so I leave this normal data here just for visualization
    # data = np.random.normal(0, 3, 1000)
    
    mean = sum(data) / len(data)
    
    x_values = [int((num - mean)) for num in data]
    
    y_values = []
    assigned_y = {}

    for x in x_values:
        if x in assigned_y:
            y = assigned_y[x] + np.random.uniform(0.05, 0.25)  # assign slightly higher y value if x is already assigned
        else:
            y = 0  # assign 0 if x is not already assigned
        y_values.append(y)
        assigned_y[x] = y  # add (x,y) pair to the dictionary
    
    # Distort the x so it does not look like perfect columns
    x_values = [(num + np.random.uniform(0.25, 1)) for num in x_values]
    
    # Create the bubble plot
    sb.scatterplot(x=x_values, y=y_values, size=data, hue=data, sizes=(10, 500), palette='rainbow', legend=False)
    
    plt.show()

@tictoc
def gap_length(files):
    """This function gets several basic statistic on the number of gaps per station and row.
    ----------
    Arguments:
    files (list) -- the numbers of the station to analyze
    
    Return:
    None"""
    
    gaps = pd.DataFrame(columns=['station', 'min', 'mean', 'std', 'max', 'Q1', 'Q2', 'Q3', 'P95'])
    
    for f in files:
        
        # Read the database
        df = pd.read_csv(f'data/labeled_{f}.csv', sep=',', encoding='utf-8')

        # Get the number of missing values per row in the database
        missing = df.isnull().sum(axis=1).tolist()
        
        # Get the length of each gap
        gap_lenghts = []
        gap_counter = 0
        for i in missing:
            if i != 0:
                gap_counter += 1
            
            elif i == 0 and gap_counter !=0:
                gap_lenghts.append(gap_counter)
                gap_counter = 0
        
        maximum, minimum = max(gap_lenghts), min(gap_lenghts)
        mean, std = np.mean(gap_lenghts), np.std(gap_lenghts)

        Q1, Q2, Q3 = np.percentile(gap_lenghts, 25), np.percentile(gap_lenghts, 50), np.percentile(gap_lenghts, 75)
        p95 = np.percentile(gap_lenghts, 95)
        
        gaps.loc[len(gaps.index)] = [f, minimum, mean, std, maximum, Q1, Q2, Q3, p95]
    
    # Save the results
    gaps.to_csv('data/gaps.csv', sep=',', encoding='utf-8', index=False)

@tictoc
def gap_row_pyplot(file, max_missing_values):
    """This function analyzes the number of gaps, their lenght and the number of variables affected in each case.
    ----------
    Arguments:
    files (int) -- the numbers of the station to analyze
    max_missing_values (int) -- the number of water quality variables in the database
    All have max_missing_values = 6, but 902 and 916 which have 7
    
    Return:
    None"""
    
    # Continue improving the presentation based on this https://plotly.com/python/pie-charts/
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{file}.csv', sep=',', encoding='utf-8')

    # Get the number of missing values per row in the database
    missing = df.isnull().sum(axis=1)
        
    # Count the number of rows with missing values (1, 2, 3, ..., max_missing_values)
    count = pd.DataFrame(index=pd.Index(range(0, max_missing_values+1), name='index'), columns=['missing_values', 'count'])
    
    for i in range(0, max_missing_values + 1):
        count.loc[i, 'missing_values'] = f'Rows missing {i} values'
        count.loc[i, 'count'] = (missing==i).sum()
    
    # Print the results
    print('Number of missing values per row')
    print(count)
    
    fig = px.pie(count, values='count', names='missing_values',
                color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20)
    fig.update_layout(showlegend=False, title=f'<b>Missing values per row in station {file}</b>')
    # fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    # fig.show()
    pio.write_image(fig, f'images/pie_chart_{file}.pdf', width=2.5*300, height=2.5*300, scale=1)
    
@tictoc
def label_analyzer(files):
    """This function studies the number of rows labeled as anomalies and those which have missing values but
    are also labeled as anomalous.
    ----------
    Arguments:
    files (list) -- the numbers of the station to analyze
    
    Return:
    None"""
    
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
    ax.set_xlabel('Station')
    ax.set_ylabel('Count')
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_title('Anomalies and missing anomalies by station')

    # Show the plot
    plt.show()
    
    # Save the image
    fig.savefig(f'images/anomalies_missing_anomalies.png', dpi=300)

def violins(variable_name, file_paths):
    """This function plots a violin-style plot of all variables in the data available.
    ----------
    Arguments:
    variable (string)-- name of the variable to plot
    
    Return:
    None"""
    
    # Create an empty list to store the data frames
    combined_df = pd.DataFrame()
    
    # Read each file and store the data frame in the combined dataframe
    for path in file_paths:
        df = pd.read_csv(path, sep=',', encoding='utf-8')
        if (variable_name + path[12:16]) in df.columns:
            combined_df = pd.concat([combined_df, df[variable_name + path[12:16]]], axis=1)

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(bottom=0.17) # Set the margins around the entire figure
    sb.violinplot(data=combined_df)
    label_title, label_y_axis = labeler(varname=variable_name)
    plt.xlabel('Variable')
    plt.ylabel(label_y_axis)
    plt.xticks(rotation=45) # Change the size of the x labels
    plt.title(label_title)
    
    fig.savefig(f'images/violin_{variable_name}.png', dpi=300)
    
    plt.show()

@tictoc
def multivar_plotter(station):
    """This computes a multivariate plot of each anomaly in a 
    specific database.
    ----------
    Arguments:
    None

    Return:
    None"""

    # Improve https://www.pythoncharts.com/
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8', parse_dates=['date'])
    
    # Normalize the data
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
    
    # Filter the data to select only rows where the label column has a value of 1
    df_index = df[df["label"] == 1]
    
    # Create a new column with the difference between consecutive dates
    date_diff = (df_index['date'] - df_index['date'].shift()).fillna(pd.Timedelta(minutes=15))

    # Create groups of consecutive dates
    date_group = (date_diff != pd.Timedelta(minutes=15)).cumsum()

    # Get the starting and ending indexes of each group of consecutive dates
    grouped = df.groupby(date_group)
    consecutive_dates_indexes = [(group.index[0], group.index[-1]) for _, group in grouped]
    
    # Set date as the index column
    df.set_index('date', inplace=True)
    
    # Drop the label column
    df.drop(df.columns[-1], axis=1, inplace=True)
    
    # Plot each anomaly
    counter = 1
    for i in consecutive_dates_indexes:

        fig = df.iloc[int(i[0]):int(i[1]), :].plot(figsize=(10,5))
        plt.title(f"Anomaly {counter} station {station}")
        plt.xlabel('Date')
        plt.ylabel('Standarized values')
        # plt.show()
        
        # Save the image
        fig = fig.get_figure()
        fig.subplots_adjust(bottom=0.19)
        fig.savefig(f'images/anomaly_{station}_{counter}.png', dpi=300)
        
        # Close the fig for better memory management
        plt.close(fig=fig)
        
        counter += 1


if __name__ == '__main__':
    
    # label_analyzer(files=[901, 902, 904, 905, 906, 907, 910, 916])
    
    multivar_plotter(station=904)
    
    # file_paths = ['data/labeled_901.csv', 'data/labeled_902.csv',
    #             'data/labeled_904.csv', 'data/labeled_905.csv',
    #             'data/labeled_906.csv', 'data/labeled_907.csv',
    #             'data/labeled_910.csv', 'data/labeled_916.csv']
    
    # violins(variable_name='water_temperature', file_paths=file_paths)


