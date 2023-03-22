"""Useful for changing the attributes of a model in case it is necessary, such as
drift detected of wanting to reset it to its original settings"""

"""Cloning a model"""
from river import datasets, linear_model, optim, preprocessing
from pprint import pprint

# Define a model to clone
model = (
    preprocessing.StandardScaler() |
    linear_model.LinearRegression(
        optimizer=optim.SGD(3e-2)
    )
)

for x, y in datasets.TrumpApproval():
    model.predict_one(x)
    model.learn_one(x, y)

# Extract weights of the last iteration
print(model[-1].weights)

# Clone
clone = model.clone()
print(clone[-1].weights) # There are no weights because the clone has not seen any data

# The learning rate
clone[-1].optimizer.learning_rate

# Clone the model but change a parameter
clone = model.clone({"LinearRegression": {"optimizer": optim.Adam()}})
clone[-1].optimizer

"""Mutating attributes"""
# This way the model is not reseted.
# e.g. change the 12 attribute and the optimizer's lr attribute
model.mutate({
    "LinearRegression": {
        "l2": 0.1,
        "optimizer": {
            "lr": optim.schedulers.Constant(25e-3)
        }
    }
})

print(repr(model))
print(model[-1].weights)
