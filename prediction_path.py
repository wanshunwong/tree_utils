from classifiers.sklearn import prediction_path_sklearn
from tree_path import TreePath


def import_decorator(func):
    def decorated_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError:
            return None
    return decorated_func


def _format_input(x):
    for f in [_format_list, _format_numpy, _format_pandas]:
        result = f(x)
        if result is not None:
            return result

    raise ValueError("x should be a list, numpy.ndarray or pandas.Series.")


def _format_list(x):
    if isinstance(x, list):
        return x, ["Feature_{}".format(i) for i in range(len(x))]


@import_decorator
def _format_numpy(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        x = x.flatten()
        return x, ["Feature_{}".format(i) for i in range(len(x))]


@import_decorator
def _format_pandas(x):
    import pandas as pd
    if isinstance(x, pd.Series):
        return x.values.flatten(), x.index.tolist()


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
    for f in [_classifier_sklearn]:
        result = f(classifier, x, feature_names)
        if result is not None:
            return result

    raise ValueError("{} is not supported.".format(classifier.__class__))


@import_decorator
def _classifier_sklearn(classifier, x, feature_names):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if isinstance(classifier, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return TreePath(prediction_path_sklearn(classifier, x, feature_names))

    if isinstance(classifier, (RandomForestClassifier, RandomForestRegressor)):
        paths = []
        for estimator in classifier.estimators_:
            paths.append(TreePath(prediction_path_sklearn(estimator, x, feature_names)))
        return paths
