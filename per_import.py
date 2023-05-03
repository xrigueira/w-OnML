# https://scikit-learn.org/stable/modules/ensemble.html

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

# https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('data/labeled_904_cle.csv', sep=',', encoding='utf-8')

# Convert variable columns to np.ndarray
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

feature_names = data.columns[1:-1].to_list()
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

start_time = time.time()
importances = forest.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()

model = Ridge(alpha=1e-2).fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_train, y_train,
                        n_repeats=30,
                        random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(feature_names[i],
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")