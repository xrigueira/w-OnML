"""The csv.DictReader can be used to read a CSV file and convert each row to a dict"""

from river import datasets
from pprint import pprint

dataset = datasets.Bikes()
# print(dataset)

# Display first sample
x, y = next(iter(dataset))
# pprint(x)
# pprint(y)

# Recommended way to iterate over a dataset with the built in function stream.iter_csv()
from river import stream
X_y = stream.iter_csv(dataset.path,
                    converters={'bikes': int, 'clouds': int, 'humidity': int, 'pressure': float, 'temperature': float, 'wind': float},
                    parse_dates={'moment': '%Y-%m-%d %H:%M:%S'},
                    target='bikes')
x, y = next(X_y)
print(x, y)