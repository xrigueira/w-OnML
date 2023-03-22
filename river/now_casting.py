# Now casting cosists in predicting the next
# value in a time series

from river import datasets

for x, y in datasets.AirlinePassengers():
    print(x, y)
    break

from river import compose
from river import linear_model
from river import preprocessing

def get_ordinal_date(x):
    return {'ordinal_date': x['month'].toordinal()}

model = compose.Pipeline(
    ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression())
)

# Evaluate the model
from river import metrics
from river import utils
import matplotlib.pyplot as plt

def evaluate_model(model):
    
    metric = utils.Rolling(metrics.MAE(), 12)
    
    dates = []
    y_trues = []
    y_preds = []
    
    for x, y in datasets.AirlinePassengers():
        
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        
        # Update the error metric
        metric.update(y, y_pred)
        
        # Store the true value and the prediction
        dates.append(x['month'])
        y_trues.append(y)
        y_preds.append(y_pred)
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(dates, y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Ground truth')
    ax.plot(dates, y_preds, lw=3, color='#e74c3c', alpha=0.8, label='Prediction')
    ax.legend()
    ax.set_title(metric)

evaluate_model(model=model)