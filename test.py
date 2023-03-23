import data
from pprint import pprint

dataset = data.labeled_901()
pprint(dataset)

# Check out the first sample
x, y = next(iter(dataset))
print(x)