from ast import literal_eval

from numpy import isnan

from tree_path import TreePath
from utilities import format_input, DICT_LIKE

# keys in the booster json dump
NODEID = "nodeid"
DEPTH = "depth"
SPLIT = "split"
SPLIT_CONDITION = "split_condition"
YES = "yes"
NO = "no"
MISSING = "missing"
CHILDREN = "children"
LEAF = "leaf"


def prediction_path_xgboost(tree, x):
    """Compute the decision path in an xgboost tree, and return a TreePath object.

    Args:
        tree (str): A (string of) json dump of an xgboost tree.
        x: A feature vector.

    Returns:
        A TreePath object.
    """
    tree = literal_eval(tree.replace(" ", ""))
    x = format_input(x, DICT_LIKE)
    path = []
    while LEAF not in tree:
        feature = tree[SPLIT]
        feature_value = x[feature]
        threshold = tree[SPLIT_CONDITION]
        if isnan(feature_value) or feature_value is None:
            child_id = tree[MISSING]
            sign = "?"
        else:
            sign_bool = feature_value <= threshold
            sign = "<=" if sign_bool else ">"
            child_id = tree[YES] if sign_bool else tree[NO]
        path.append((tree[DEPTH], tree[NODEID], feature, feature_value, sign, threshold, ""))

        children = tree[CHILDREN]
        for child in children:
            if child[NODEID] == child_id:
                tree = child
    else:
        path.append((len(path), tree[NODEID], "", "", "", "", tree[LEAF]))
    return TreePath(path, optional_header=["Leaf Value"])
