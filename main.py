import data
import pandas as pd
from tictoc import tictoc

class Imputator():
    
    def __init__(self, station) -> None:
        self.dataset = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8')
        self.station = station
    
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
    
    def imputation_knn(self):
        """Performs data "imputation" with the kNN method.
        ----------
        Arguments:
        station -- the number of the station to analyze
        
        Return:
        None"""

        # Split the dataframe into two parts: one with missing values, and another without them
        df_missing = self.dataset[self.dataset.isnull().any(axis=1)]
        df_not_missing = self.dataset[~self.dataset.isnull().any(axis=1)]

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
        df_dropped = self.dataset[drop_columns]
        
        df_imputed = pd.concat([df_dropped, df_imputed], axis=1)
        
        # Move 'label' column to the last possition
        col_to_move = df_imputed.pop('label')
        df_imputed.insert(len(df_imputed.columns), 'label', col_to_move)

        # Save the new dataframe
        df_imputed.to_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8', index=False)


class Model():

    def __init__(self) -> None:
        self.dataset = data.labeled_901()

    @tictoc
    def logreg(self):
        """This method performs logist regression.
        ----------
        Arguments:
        self.database = loaded data.

        Return:
        None"""

        from river import compose
        from river import metrics
        from river import linear_model
        # from river import preprocessing

        model = compose.Pipeline(
            compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
            # Add the standard scaler and see if it works: preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )

        # Documentation on ROC AUC: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            model.learn_one(x, y)
            metric.update(y, y_pred)
            # print(model.debug_one(x))

        print(metric)

    @tictoc
    def hoefftree(self):
        """This method performs classification with Hoeffding trees.
        ----------
        Arguments: 
        self.dataset: loaded dataset
        
        Return:
        None"""

        from river import tree
        from river import compose
        from river import metrics
        from river import evaluate

        model = compose.Pipeline(
            compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
            tree.HoeffdingTreeClassifier(grace_period=200)
        )

        metric = metrics.Accuracy()

        evaluate.progressive_val_score(self.dataset, model, metric)

        print(metric)

    @tictoc
    def halfspace(self):
        """This method performs anomaly detection with the half-space trees algorithm.
        Half-space trees is an online variant of isolation forests. They work well when 
        anomalies are spread out. However, they do not work well if anomalies are 
        packed together in windows (which is our case).
        ----------
        Arguments: 
        self.dataset: loaded dataset
        
        Return:
        None"""

        from river import anomaly
        from river import compose
        from river import metrics
        from river import preprocessing

        model = compose.Pipeline(
            compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(seed=24)
        )

        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            score = model.score_one(x)
            model = model.learn_one(x)
            metric = metric.update(y, score)

        print(metric)

    @tictoc
    def logref_imb(self):
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
            compose.Select("year", "month", "day", "hour", "minute", "second", "week", "weekOrder", "ammonium_901", "conductivity_901", "dissolved_oxygen_901", "ph_901", "turbidity_901", "water_temperature_901"),
            imblearn.RandomSampler(
                classifier=linear_model.LogisticRegression(),
                desired_dist={0: .8, 1: .2},                    # Samples data to contain 80% of 0s and 20% of 1s
                sampling_rate=.01,                              # Trains with 1% of the data
                seed=42
            )
        )
        
        metric = metrics.ROCAUC()

        for x, y in self.dataset:
            y_pred = model.predict_proba_one(x)
            model.learn_one(x, y)
            metric.update(y, y_pred)
            # print(model.debug_one(x))

        print(metric)

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

if __name__ == '__main__':

    station = 901
    
    # Impute the data
    imputator = Imputator(station=901)
    imputator.imputation_del()
    
    # # Call the model
    # model = Model()
    # model.hoefftree()