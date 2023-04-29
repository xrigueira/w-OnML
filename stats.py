import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    dataframe = dataframe.iloc[:, 1:-1]
    
    # Get the correlation matrix
    correlation_matrix = dataframe.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Set the margins around the entire figure
    fig.subplots_adjust(left=0.165)
    
    sb.heatmap(correlation_matrix, cmap='RdBu', annot=True, annot_kws={"fontsize":10}, cbar=True, square=True, ax=ax, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
    plt.xticks(rotation=45, fontsize=10) # Change the size of the x labels
    plt.yticks(rotation=45, fontsize=10) # Change the size of the y labels
    plt.title(f'Correlation matrix station {station}')
    
    fig.savefig(f'images/corr_{station}.png', dpi=300)
    
    # plt.show()

def biplot(score, coef, station, labels=None):

    xs = score[:,0]
    ys = score[:,1]
    n = coef.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,
                s=5, 
                color='cornflowerblue')

    for i in range(n):
        plt.arrow(0, 0, coef[i,0], 
                coef[i,1], color = 'black',
                alpha = 0.5)
        plt.text(coef[i,0]* 1.05, 
                coef[i,1] * 1.05, 
                labels[i], 
                color = 'darkblue', 
                ha = 'center', 
                va = 'center')

    plt.title(f'Biplot station {station}')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))    

    # plt.figure()

def pca(dataframe, station):
    """Performs principal components analysis.
    ----------
    Arguments:
    dataframe (pandas.dataframe): contains the data
    station (int): the station number
    
    Returns:
    None"""
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Separate the data and label columns
    X = StandardScaler().fit_transform(dataframe.iloc[:, 1:-1])
    y = dataframe.iloc[:, -1]

    # Peform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Get the explained variance ratio of the principal components
    explained_var = pca.explained_variance_ratio_

    # Print the variance explained by each component
    print(f'Explained variance ratio: {explained_var}')

    # # Bar plot of explained_variance
    # plt.bar(
    #     range(1,len(pca.explained_variance_)+1),
    #     pca.explained_variance_,
    #     color='blue'
    #     )
    
    # plt.plot(
    #     range(1,len(pca.explained_variance_ )+1),
    #     np.cumsum(pca.explained_variance_),
    #     c='red',
    #     label='Cumulative Explained Variance')
    
    # plt.legend(loc='upper left')
    # plt.xlabel('Number of components')
    # plt.ylabel('Explained variance (eignenvalues)')
    # plt.title(f'Scree plot station {station}')
    # plt.show()

    # Biplot
    pca_dataframe = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    
    plt.title('Biplot of PCA')
    biplot(X_pca, np.transpose(pca.components_), station, list(dataframe.iloc[:, 1:-1].columns))
    
    plt.savefig(f'images/biplot_{station}.png', dpi=300)

    # plt.show()
    plt.close('all')

if __name__ == '__main__':
    
    # Define the station number
    stations = [901, 902, 904, 905, 906, 907, 910, 916]
    
    for station in stations:
    
        # Read the data
        df = pd.read_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8')
    
        # correlation(dataframe=df, station=station)

        pca(dataframe=df, station=station)