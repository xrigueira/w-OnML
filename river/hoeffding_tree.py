import datetime as dt
import matplotlib.pyplot as plt

from river import tree
from river import metrics
from river import datasets
from river import evaluate
from river import preprocessing
from river.datasets import synth
from pprint import pprint
"""How to inspect tree models?"""
dataset = datasets.Phishing()
print(dataset)

# Train an instance of the HoeffdingTreeClassifier using this dataset
model = tree.HoeffdingTreeClassifier()

for x, y in dataset:
    model.learn_one(x, y)

pprint(model)
pprint(model.summary)

# Check on how it predicts on a specific instance with debug_one
x, y = next(iter(dataset))
pprint(x)
pprint(y)

print(model.debug_one(x))

# Limit memory usage
def plot_performance(dataset, metric, models):
    metric_name = metric.__class__.__name__

    # Make the generated data reusable
    dataset = list(dataset)
    fig, ax = plt.subplots(figsize=(10, 5), nrows=3, dpi=300)
    for model_name, model in models.items():
        step = []
        error = []
        r_time = []
        memory = []

        for checkpoint in evaluate.iter_progressive_val_score(
            dataset, model, metric, measure_time=True, measure_memory=True, step=100
        ):
            step.append(checkpoint["Step"])
            error.append(checkpoint[metric_name].get())

            # Convert timedelta object into seconds
            r_time.append(checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory = checkpoint["Memory"]
            memory.append(raw_memory * 2**-20)

        ax[0].plot(step, error, label=model_name)
        ax[1].plot(step, r_time, label=model_name)
        ax[2].plot(step, memory, label=model_name)

        ax[0].set_ylabel(metric_name)
        ax[1].set_ylabel('Time (seconds)')
        ax[2].set_ylabel('Memory (MB)')
        ax[2].set_xlabel('Instances')

        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        ax[0].legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.25),
            ncol=3, fancybox=True, shadow=True
        )

        plt.tight_layout()
        plt.close()

        return fig

plot_performance(
    synth.Friedman(seed=42).take(10_000),
    metrics.MAE(),
    {
        "Unbounded HTR": (
            preprocessing.StandardScaler() |
            tree.HoeffdingTreeRegressor(splitter=tree.splitter.EBSTSplitter())
        )
    }
)