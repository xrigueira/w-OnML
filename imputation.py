
import pandas as pd

"""This file contain four functions for data imputation."""

# The selected method has to replace mfilterer.py in the preprocessing file

# Data imputation methods
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

def imputation_knn2(station):
    """Performs data "imputation" with the kNN method.
    ----------
    Arguments:
    station -- the number of the station to analyze
    
    Return:
    None"""
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')

    # Split the dataframe into two parts: one with missing values, and another without them
    df_missing = df[df.isnull().any(axis=1)]
    df_not_missing = df[~df.isnull().any(axis=1)]

    # Drop the nonvariable columns
    drop_columns = ['date', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'weekOrder', 'label']

    df_missing = df_missing.drop(drop_columns, axis=1)
    df_not_missing = df_not_missing.drop(drop_columns, axis=1)

    # Normalize the data
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_not_missing_normalized = pd.DataFrame(scaler.fit_transform(df_not_missing), columns=df_not_missing.columns)
    df_missing_normalized = pd.DataFrame(scaler.transform(df_missing), columns=df_missing.columns, index=df_missing.index)

    # Use kNN imputation to fill in the missing values
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)
    df_missing_imputed = pd.DataFrame(imputer.fit_transform(df_missing_normalized), columns=df_missing_normalized.columns, index=df_missing.index)

    # Inverse normalize the data (maybe this step is not needed)
    df_missing_imputed = pd.DataFrame(scaler.inverse_transform(df_missing_imputed), columns=df_missing.columns, index=df_missing.index)

    # Add the missing rows back to the original dataframe
    df_imputed = pd.concat([df_not_missing, df_missing_imputed], axis=0)

    # Add the dropped columns
    df_dropped = df[drop_columns]
    
    df_imputed = pd.concat([df_dropped, df_imputed], axis=1)
    
    # Move 'label' column to the last possition
    col_to_move = df_imputed.pop('label')
    df_imputed.insert(len(df_imputed.columns), 'label', col_to_move)

    # Save the new dataframe
    df_imputed.to_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', index=False)

def imputation_svm(station):
    """_summary_

    Args:
        station (_type_): _description_
    """
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
    
    # TODO: see GPT answer

def imputation_logreg(station):
    """_summary_

    Args:
        station (_type_): _description_
    """
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
    
    # TODO: look up and ask GPT
    
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

if __name__ == '__main__':

    imputation_knn2(station=901)