"""Active learning is a training regime, where the goal is to fit a model on 
the most discriminative samples. It is usually applied in situations where a 
limited amount of labeled data is available."""

from river import active
from river import metrics
from river import datasets
from river import linear_model
from river import feature_extraction

dataset = datasets.SMSSpam()
metric = metrics.Accuracy()
model = (
    feature_extraction.TFIDF(on='body') |
    linear_model.LogisticRegression()
)
model = active.EntropySampler(model, seed=42)

n_samples_used = 0
for x, y in dataset:
    y_pred, ask = model.predict_one(x)
    metric.update(y, y_pred)
    if ask:
        n_samples_used += 1
        model.learn_one(x, y)

print(metric)

print(f"{n_samples_used} / {dataset.n_samples} = {n_samples_used / dataset.n_samples:.2%}")

from river import evaluate

evaluate.progressive_val_score(
    dataset=dataset,
    model=model.close(),
    metric=metric.clone(),
    print_every=1000
)