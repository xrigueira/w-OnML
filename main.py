import data
import numpy as np
import pandas as pd
from tictoc import tictoc

class Imputator():
    
    def __init__(self, station) -> None:
        self.dataset = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
        self.station = station
    
    def selector(self):
        """This function returns the column names of
        the database as they will be required by the Model()
        ---------
        Arguments:
        self
        
        Return:
        Column names but the first and last one"""
        
        return list(self.dataset.columns[1:-1])
    
    @tictoc
    def imputation_del(self):
        """Performs data "imputation" by deleting all those rows with missing values.
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""
        
        # Remove all rows with misssing
        self.dataset = self.dataset.dropna()
        
        # Save the new dataframe
        self.dataset = self.dataset.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)
    
    @tictoc
    def imputation_iter(self):
        """Performs data imputation by iterating on all those rows with missing values.
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""
        
        # Iterate
        self.dataset = (self.dataset.interpolate(method='polynomial', order=1)).round(2)
        
        # Save the new dataframe
        self.dataset.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)
    
    @tictoc
    def imputation_knn(self):
        """Performs data imputation with the kNN method.
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""

        # Split the dataframe into two parts: one with missing values, and another without them
        df_missing = self.dataset[self.dataset.isnull().any(axis=1)]
        df_not_missing = self.dataset[~self.dataset.isnull().any(axis=1)]

        # Drop the nonvariable columns
        drop_columns = ['date', 'label']

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
        df_dropped = self.dataset[drop_columns]
        
        df_imputed = pd.concat([df_dropped, df_imputed], axis=1)
        
        # Move 'label' column to the last possition
        col_to_move = df_imputed.pop('label')
        df_imputed.insert(len(df_imputed.columns), 'label', col_to_move)

        # Save the new dataframe
        df_imputed.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)

    @tictoc
    def imputation_svm(self):
        """Performs data imputation with the kNN method
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""
        
        # Split the dataframe into two parts: one with missing values, and another without them
        df_missing = self.dataset[self.dataset.isnull().any(axis=1)]
        df_not_missing = self.dataset[~self.dataset.isnull().any(axis=1)]
        
        # Drop the 'date' column as it is not an integer float and the label
        df_missing = df_missing.drop(['date', 'label'], axis=1)
        df_not_missing = df_not_missing.drop(['date', 'label'], axis=1)
        
        # Split the dataframe without missing values into features (X) and targets (y) variables.
        # In this case, the target variables in the columns with missing values
        variables = list(self.dataset.columns[1:-1])
        X = df_not_missing.drop(variables, axis=1)
        y = df_not_missing[variables]
        
        # Split the data into training and testing samples
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_test = X_test[:len(df_missing)]
        
        # Train a Support Vector Regression model on the training set
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        
        model = MultiOutputRegressor(SVR(kernel='rbf', C=10, gamma=0.1))
        model.fit(X_train, y_train)
        
        # Use the trained model to predict the missing values
        predicted = model.predict(X_test)
        
        # Replace the missing values in the original dataframe with the predicted values
        df_missing.loc[df_missing[variables].index, variables] = pd.DataFrame(predicted, index=df_missing[variables].index, columns=variables)

        # Merge the dataframe with imputed values with the dataframe without missing values
        df_imputed = pd.concat([df_missing, df_not_missing])

        # Insert the original 'date' and 'label' columns
        date_col = self.dataset.pop('date')
        label_col = self.dataset.pop('label')
        df_imputed.insert(0, 'date', date_col)
        df_imputed.insert(len(df_imputed.columns), 'label', label_col)
        
        # Save the new dataframe
        df_imputed.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)

    @tictoc
    def imputation_linreg(self):
        """Performs imputation using linear regression
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""
        
        # Split the dataframe into two parts: one with missing values, and another without them
        df_missing = self.dataset[self.dataset.isnull().any(axis=1)]
        df_not_missing = self.dataset[~self.dataset.isnull().any(axis=1)]
        
        # Drop the 'date' column as it is not an integer or float
        df_missing = df_missing.drop(['date'], axis=1)
        df_not_missing = df_not_missing.drop(['date',], axis=1)
        
        # Train a linear regression model on the dataframe without missing values
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        
        variables = list(self.dataset.columns[1:-1])
        X_train = df_not_missing.drop(variables, axis=1)
        y_train = df_not_missing[variables]
        
        model.fit(X_train, y_train)
        
        # Use the trained model to predict the values in the dataframe with missing values
        X_test = df_missing.drop(variables, axis=1)
        y_hat = model.predict(X_test)
        
        # Replace the missing values in the original dataframe with the predicted ones
        self.dataset.loc[self.dataset.isnull().any(axis=1), variables] = y_hat
        
        # Save the new dataframe
        self.dataset.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)
    
    @tictoc
    def imputation_trees(self):
        """Performs data imputation with the missForest algorithm.
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""
        
        from missingpy import MissForest
        
        # Identify the columns with missing values
        columns_with_missing_values = self.dataset.columns[self.dataset.isnull().any()].tolist()
        
        # Create a separate dataframe with only the columns with missing values
        df_missing = self.dataset[columns_with_missing_values]
        
        # Create an instance of the missForest algorithm and impute missing values
        imputer = MissForest(criterion='squared_error')
        imputed_df_missing = imputer.fit_transform(df_missing)
        
        # Replace missing values in the original dataframe with imputed values
        self.dataset[columns_with_missing_values] = imputed_df_missing
        
        # Save the new dataframe
        self.dataset.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)

class Model():

    def __init__(self, station, columns) -> None:
        self.station = station
        self.columns = columns
        if self.station == 901:
            self.dataset = data.labeled.labeled_901()
        elif self.station == 902:
            self.dataset = data.labeled.labeled_902()
        elif self.station == 904:
            self.dataset = data.labeled.labeled_904()
        elif self.station == 905:
            self.dataset = data.labeled.labeled_905()
        elif self.station == 906:
            self.dataset = data.labeled.labeled_906()
        elif self.station == 907:
            self.dataset = data.labeled.labeled_907()
        elif self.station == 910:
            self.dataset = data.labeled.labeled_910()
        elif self.station == 916:
            self.dataset = data.labeled.labeled_916()

    def get_labels(self):
        """This function returns the column labels of
        the database as it will be required by Metric()
        ---------
        Arguments:
        self
        
        Return:
        Column labels"""
        
        data = pd.read_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8')
        
        return list(data['label'])

    @tictoc
    def logreg(self):
        """This method performs logist regression.
        Fast and good results.
        ----------
        Arguments:
        self.database = loaded data.

        Return:
        y_preds (list): contains the predictions."""
        
        from river import compose
        from river import metrics
        from river import linear_model
        from river import preprocessing
        
        # Convert dictionary keys to a list of column names
        
        model = compose.Pipeline(
            compose.Select(*self.columns),
            # Add the standard scaler and see if it works: preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )

        # Documentation on ROC AUC: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
        y_preds = []
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)

        print(metric)
        
        metric = metric.get() # In case we want to return it as a float
        
        return y_preds

    @tictoc
    def hoefftree(self):
        """This method performs classification with Hoeffding trees.
        Good results.
        ----------
        Arguments: 
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        y_preds (list): contains the predictions"""

        from river import tree
        from river import compose
        from river import metrics

        model = compose.Pipeline(
            compose.Select(*self.columns),
            tree.HoeffdingTreeClassifier(grace_period=200)
        )

        y_preds = []
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)

        print(metric)

        return y_preds

    @tictoc
    def halfspace(self):
        """This method performs anomaly detection with the half-space trees algorithm.
        Half-space trees is an online variant of isolation forests. They work well when 
        anomalies are spread out. However, they do not work well if anomalies are 
        packed together in windows (which is our case).
        ----------
        Arguments: 
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        y_preds (list): contains the predictions"""

        from river import anomaly
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select(*self.columns),
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(seed=24)
        )

        y_preds = []
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            score = model.score_one(x)
            y_preds.append(score)
            model = model.learn_one(x)
            metric = metric.update(y, score)

        print(metric)
        
        return y_preds

    @tictoc
    def oneclasssvm(self):
        """This method performs anomaly detection with a one class SVM.
        The results are not that great probably due to the complexity of
        the problem. Most likely, the frontiers learn by the SVM are not
        good enough.
        ----------
        Arguments: 
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        None"""

        from river import anomaly
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select(*self.columns),
            preprocessing.StandardScaler(),
            anomaly.QuantileFilter(
                anomaly.OneClassSVM(),
                q=0.98
            )
        )

        y_preds = []
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            score = model.score_one(x)
            is_anomaly = model['QuantileFilter'].classify(score)
            y_preds.append(is_anomaly)
            model = model.learn_one(x)
            metric = metric.update(y, is_anomaly)

        print(metric)
        np.save('y_preds.npy', y_preds)
        return y_preds

    @tictoc
    def amfclassifier(self):
        """This method implements the Aggregated Mondrian Forest
        classifier. The results were good, but it is too slow.
        ----------
        Arguments
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        None"""

        from river import forest
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select(*self.columns),
            preprocessing.StandardScaler(),
            forest.AMFClassifier(
                n_estimators=10,
                use_aggregation=True,
                dirichlet=0.5,
                seed=1
            )
        )

        y_preds = []
        metric = metrics.ROCAUC()
        
        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)

        print(metric)
        np.save('y_preds.npy', y_preds)
        return y_preds

    @tictoc
    def arfclassifier(self):
        """This method implements the adaptative random forest classifier.
        Good results.
        ----------
        Arguments
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        None
        """

        from river import forest
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select(*self.columns),
            preprocessing.StandardScaler(),
            forest.ARFClassifier(seed=1)
        )

        y_preds = []
        metric = metrics.ROCAUC()
        
        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)

        print(metric)
        
        return y_preds

    @tictoc
    def fastdecisiontree(self):
        """This method implements the Extremely Fast Decision Tree 
        classifier. Also refered to as the Hoeffding AnyTime Tree (HATT).
        Good results.
        ----------
        Arguments:
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        None"""

        from river import tree
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select(*self.columns),
            preprocessing.StandardScaler(),
            tree.ExtremelyFastDecisionTreeClassifier(grace_period=1000,
                delta=1e-5,
                min_samples_reevaluate=1000)
        )

        y_preds = []
        metric = metrics.ROCAUC()
        
        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)

        print(metric)
        
        return y_preds

    @tictoc
    def sgt(self):
        """This method implements stochastic gradient tree for 
        binary classification.
        ----------
        Arguments:
        self.dataset (pd.dataframe): loaded dataset
        
        Return:
        None"""

        from river import tree
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select(*self.columns),
            preprocessing.StandardScaler(),
            tree.SGTClassifier(
                feature_quantizer=tree.splitter.StaticQuantizer(
                n_bins=32, warm_start=10
                )
            )
        )

        y_preds = []
        metric = metrics.ROCAUC()
        
        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)

        print(metric)
        
        return y_preds

    @tictoc
    def logreg_imb(self):
        """This method implements logistic regression with imbalanced data.
        This means that the data has a lot more 0s than 1s, therefore we try
        to balance this out.
        ----------
        Arguments: 
        self.dataset: loaded dataset
        
        Return:
        None"""

        import collections

        from river import compose
        from river import metrics
        from river import imblearn
        from river import linear_model

        counts = collections.Counter(y for _, y in self.dataset)

        for c, count in counts.items():
            print(f'{c}: {count} ({count / sum(counts.values()):.5%})')
        
        model = compose.Pipeline(
            compose.Select(*self.columns),
            imblearn.RandomSampler(
                classifier=linear_model.LogisticRegression(),
                desired_dist={0: .7, 1: .3},                    # Samples data to contain 80% of 0s and 20% of 1s
                sampling_rate=.01,                              # Trains with 1% of the data
                seed=42
            )
        )
        
        y_preds = []
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            y_preds.append(y_pred)
            model.learn_one(x, y)
            metric.update(y, y_pred)
        
        print(metric)
        
        return y_preds

    @tictoc
    def logreg_many(self):
        """This method performs logistic regression, but with learn and 
        predict many. Very bad resutls.
        ----------
        Arguments:
        self.database = loaded data.

        Return:
        y_preds (list): contains the predictions."""
        
        from river import compose
        from river import metrics
        from river import linear_model
        from river import preprocessing
        from itertools import islice
        
        # Convert dictionary keys to a list of column names
        model = compose.Pipeline(
            compose.Select(*self.columns),
            # Add the standard scaler and see if it works: preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )

        # Initialize a list to store the predicted probabilities
        y_preds = []

        # Train the model on the first 1000 samples of the dataset using learn_one()
        for x, y in islice(self.dataset, 1000):
            model.learn_one(x, y)

        # Train the model on the rest of the dataset using learn_many()
        X_train, y_train = zip(*[(x, y) for x, y in islice(self.dataset, 1000, None)])
        X_train, y_train = pd.DataFrame(X_train), pd.Series(y_train)
        model.learn_many(X_train, y_train)

        # Make predictions on the entire dataset using predict_many()
        X_test, y_test = zip(*[(x, y) for x, y in self.dataset])
        X_test = pd.DataFrame(X_test)
        y_preds = model.predict_many(X_test)
        
        return y_preds
    
    @tictoc
    def adwin(self):
        """This method implements drift detection with ADaptative WINdowing (ADWIN). 
        ----------
        Arguments:
        None

        Returns:
        None"""

        import pandas as pd
        from river import drift

        # Read the database with pandas
        df = pd.read_csv(f'data/labeled_901_cle.csv', sep=',', encoding='utf-8')

        adwin = drift.ADWIN(delta=0.002)
        drift = []

        for i, val in enumerate(df.conductivity_901):                       # The variable selected is defined manually for now
            in_drift, in_warning = adwin.update(val)
            if in_drift:
                # print(f'Drift detected at index {i}, input value: {val}')
                drift.append(i)

        print(len(drift))

class Metric():
    
    def __init__(self, labels, predicted_labels, model_used, anomaly_tail) -> None:
        self.labels = labels
        self.anomaly_tail = anomaly_tail
        self.model_used = model_used
        if model_used == 'halfspace':
            self.predicted_labels = [0 if i >= 0.5 else 1 for i in predicted_labels]
        elif model_used == 'oneclasssvm':
            self.predicted_labels = [int(i) for i in predicted_labels]
        elif model_used == 'hoefftree' or 'amfclassifier' or 'arfclassifier' or 'fastdecisiontree' or 'sgt':
            self.predicted_labels = [0 if (len(i)==0) or (i[0] >= 0.5) else 1 for i in predicted_labels]
        else:
            self.predicted_labels = [0 if i[False] >= 0.5 else 1 for i in predicted_labels]
    
    # Get the start end end index of each anomaly in labels
    def find_anomalies(self):
        """This function gets the start and end index of each
        anomaly in labels
        ----------
        Arguments:
        self
        
        Returns:
        anomalies (list): list of tuples that contain the index of the beginning and end
        of all anomalies"""
        
        anomalies = []
        i = 0
        while i < len(self.labels):
            if self.labels[i] == 1:
                j = i + 1
                while j < len(self.labels) and self.labels[j] == 1:
                    j += 1
                if j < len(self.labels) and self.labels[j] == 0:
                    anomalies.append((i, j))
                i = j
            else:
                i += 1
        
        return anomalies
    
    def match_percentage(self):
        """This function gets the match  percentage between the anomalies and the next len(anomaly) items.
        ----------
        Arguments:
        self
        
        Returns:
        anomalies (float): matcch between labels and predicted labels"""
        
        anomalies = self.find_anomalies()
        total_matches = 0
        total_items = 0
        for start, end in anomalies:
            anomaly_length = end - start
            segment = self.predicted_labels[start:end+int(anomaly_length*self.anomaly_tail)]
            total_matches += sum(a == b for a, b in zip(self.labels[start:end+int(anomaly_length*self.anomaly_tail)], segment))
            total_items += (anomaly_length + int(anomaly_length*self.anomaly_tail))
        
        return total_matches / total_items if total_items > 0 else 1.0

if __name__ == '__main__':

    station = 904
    
    # Impute the data
    imputator = Imputator(station=station)
    columns = imputator.selector()
    imputator.imputation_del()
    
    # Call the model
    model = Model(station=station, columns=columns)
    labels = model.get_labels()
    y_preds = model.logreg_imb()
    
    # Call the custom metric and get the result
    metric = Metric(labels=labels, predicted_labels=y_preds, model_used=model.logreg.__name__, anomaly_tail=0.25)
    result = metric.match_percentage()
    
    print('Metric result:', result)