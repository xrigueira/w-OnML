
import pandas as pd

"""This file contain four functions for data imputation."""

# The selected method has to replace mfilterer.py in the preprocessing file

# Data imputation
def imputation_del(station):
    """Performs data "imputation" by deleting all those rows with missing values.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""

    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')

    # Remove all rows with missing values
    df = df.dropna()

    # Save the new dataframe
    df.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)

def imputation_iter(station):
    """Performs data imputation by iterating on all those rows with missing values.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')

    # Iterate
    df = (df.interpolate(method='polynomial', order=1)).round(2)
    
    # Save the new dataframe
    df.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)

def imputation_knn(station):
    """Performs data "imputation" with the kNN method.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    
    # See if it gives memory error in the pc
    import sys
    import sklearn.neighbors._base
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
    
    from missingpy import KNNImputer
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
    
    # Identify the columns with missing values
    columns_with_missing_values = cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    # Create a separate dataframe with only the columns with missing values
    df_missing = df[cols_with_missing]
    
    # Create an instance of the missForest algorithm and impute missing values
    imputer = KNNImputer()
    imputed_df_missing = imputer.fit_transform(df_missing)
    
    # Replace missing values in the original dataframe with imputed values
    df[cols_with_missing] = imputed_df_missing
    
    # Save the new dataframe
    df.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)

def imputation_trees(station):
    """Performs data "imputation" with the missForest algorithm.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    
    import sys
    import sklearn.neighbors._base
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
    
    from missingpy import MissForest
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
    
    # Identify the columns with missing values
    columns_with_missing_values = cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    # Create a separate dataframe with only the columns with missing values
    df_missing = df[cols_with_missing]
    
    # Create an instance of the missForest algorithm and impute missing values
    imputer = MissForest(criterion='squared_error')
    imputed_df_missing = imputer.fit_transform(df_missing)
    
    # Replace missing values in the original dataframe with imputed values
    df[cols_with_missing] = imputed_df_missing
    
    # Save the new dataframe
    df.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)