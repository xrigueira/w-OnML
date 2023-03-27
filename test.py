import pandas as pd

import matplotlib.pyplot as plt

# Read the database
gaps = pd.read_csv(f'data/gaps.csv', sep=',', encoding='utf-8')

plt.hist(gaps.values, alpha=0.80, label=gaps.columns)

# set the title and y-axis label of the plot
plt.title('Boxplot of columns')
plt.ylabel('Value')
plt.legend()

# show the plot
plt.show()

