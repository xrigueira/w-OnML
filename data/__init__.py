"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you
are interested in infinite synthetic data generators.

"""
from .labeled_901 import labeled_901
from .labeled_902 import labeled_902
from .labeled_904 import labeled_904
from .labeled_905 import labeled_905
from .labeled_906 import labeled_906
from .labeled_907 import labeled_907
from .labeled_910 import labeled_910
from .labeled_916 import labeled_916

__all__ = [
    "labeled_901",
    "labeled_902",
    "labeled_904",
    "labeled_905",
    "labeled_906",
    "labeled_907",
    "labeled_910",
    "labeled_916"
]
