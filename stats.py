import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

"""This file includes several functions analyze the data"""

def correlation(dataframe, station):
    """Gets the correlation matrix.
    ----------
    Arguments:
    dataframe (pandas.dataframe): contains the data
    station (int): the station number
    
    Returns:
    None"""    
    
    # Select the columns which contain actual environmental data
    dataframe = dataframe.iloc[:, 9:-1]
    
    # Get the correlation matrix
    correlation_matrix = dataframe.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    sb.heatmap(correlation_matrix, cmap='RdBu', annot=True, annot_kws={"fontsize":8}, cbar=True, square=True, ax=ax, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
    plt.xticks(rotation=90, fontsize=7) # Change the size of the x labels
    plt.yticks(fontsize=7)              # Change the size of the y labels
    plt.title(f'Correlation matrix station {station}')
    plt.show()

def pca(dataframe, station):
    """Performs principal components analysis.
    ----------
    Arguments:
    dataframe (pandas.dataframe): contains the data
    station (int): the station number
    
    Returns:
    None"""
    
    # Look up how to do this
    # https://www.google.com/search?q=pca+analysis+python&rlz=1C1FKPE_esES940ES940&oq=pca+analysis+python&aqs=chrome..69i57.3892j0j15&sourceid=chrome&ie=UTF-8
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

if __name__ == '__main__':
    
    # Define the station number
    station = 901
    
    # Read the data
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
    
    correlation(dataframe=df, station=station)