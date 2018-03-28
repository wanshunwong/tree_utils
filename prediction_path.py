from classifiers.lightgbm import prediction_path_lightgbm
from classifiers.sklearn import prediction_path_sklearn
from classifiers.xgboost import prediction_path_xgboost
from utilities import import_decorator


def prediction_path(classifier, x, tree_index=None):
    """Compute and return the prediction path

    Args:
        classifier: A tree-based or forest-based machine learning classifier.
        x (list, numpy.ndarray, or pandas.Series): A feature vector to compute the prediction path for.
        tree_index (int or None): If the classifier is an ensemble of trees, it is the index of a target tree to
            compute the prediction path. If tree_index is set to None, all the trees in the ensemble will be used.

    Returns:
        A TreePath or a list of TreePath's.

    Raises:
        TypeError: If the class of classifier is not supported.
    """
    for f in [_classifier_sklearn, _classifier_xgboost, _classifer_lightgbm]:
        result = f(classifier, x, tree_index)
        if result is not None:
            return result

    raise TypeError("{} is not supported.".format(classifier.__class__))


def _ensemble(ensemble, x, tree_index, path_func):
    if tree_index is not None:
        return path_func(ensemble[tree_index], x)
    else:
        paths = []
        for tree in ensemble:
            paths.append(path_func(tree, x))
        return paths


@import_decorator
def _classifier_sklearn(classifier, x, tree_index):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if isinstance(classifier, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return prediction_path_sklearn(classifier, x)

    if isinstance(classifier, (RandomForestClassifier, RandomForestRegressor)):
        return _ensemble(classifier, x, tree_index, prediction_path_sklearn)


@import_decorator
def _classifier_xgboost(classifier, x, tree_index):
    from xgboost import Booster, XGBModel

    if isinstance(classifier, XGBModel):
        classifier = classifier.get_booster()

    if isinstance(classifier, Booster):  # assuming the booster is either gbtree or dart
        ensemble = classifier.get_dump(dump_format="json")
        return _ensemble(ensemble, x, tree_index, prediction_path_xgboost)


@import_decorator
def _classifer_lightgbm(classifier, x, tree_index):
    from lightgbm import LGBMClassifier, LGBMRegressor, Booster
    if isinstance(classifier, (LGBMClassifier, LGBMRegressor)):
        classifier = classifier.booster_

    if isinstance(classifier, Booster):
        ensemble = classifier.dump_model()["tree_info"]
        return _ensemble(ensemble, x, tree_index, prediction_path_lightgbm)
