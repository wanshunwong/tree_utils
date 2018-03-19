import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from classifiers.sklearn import prediction_path_sklearn
from tree_path import TreePath


def _format_input(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()
        return x, ["Feature_{}".format(i) for i in range(len(x))]
    elif isinstance(x, pd.Series):
        return x.values.flatten(), x.index.tolist()
    else:
        raise ValueError("x should be an numpy.ndarray or a pandas.Series.")


def prediction_path(classifier, x):
    """Compute and return the prediction path

    Args:
        classifier: A tree-based or forest-based machine learning classifier.
        x: A numpy array or a row of a pandas.DataFrame, i.e. a pandas.Series.

    Returns:
        A TreePath or a list of TreePath's, corresponding to a tree-based or forest-based classifier respectively.

    Raises:
        ValueError: If the class of classifier is not supported.
    """
    x, feature_names = _format_input(x)

    if isinstance(classifier, DecisionTreeClassifier) or isinstance(classifier, DecisionTreeRegressor):
        return TreePath(prediction_path_sklearn(classifier, x, feature_names))

    if isinstance(classifier, RandomForestClassifier) or isinstance(classifier, RandomForestRegressor):
        paths = []
        for estimator in classifier.estimators_:
            paths.append(TreePath(prediction_path_sklearn(estimator, x, feature_names)))
        return paths

    raise ValueError("{} is not supported.".format(classifier.__class__))
