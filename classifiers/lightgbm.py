from numpy import isnan

from tree_path import TreePath
from utilities import format_input, LIST_LIKE

# keys in the booster json dump
TREE_STRUCTURE = "tree_structure"
DECISION_TYPE = "decision_type"
DEFAULT_LEFT = "default_left"
INTERNAL_COUNT = "internal_count"
INTERNAL_VALUE = "internal_value"
LEFT_CHILD = "left_child"
MISSING_TYPE = "missing_type"
RIGHT_CHILD = "right_child"
SPLIT_FEATURE = "split_feature"
SPLIT_GAIN = "split_gain"
SPLIT_INDEX = "split_index"
THRESHOLD = "threshold"
LEAF_COUNT = "leaf_count"
LEAF_INDEX = "leaf_index"
LEAF_VALUE = "leaf_value"


def prediction_path_lightgbm(tree, x):
    tree = tree[TREE_STRUCTURE]
    x, feature_names = format_input(x, LIST_LIKE)
    path = []
    depth = 0

    while LEAF_INDEX not in tree:
        feature_id = tree[SPLIT_FEATURE]
        threshold = tree[THRESHOLD]
        feature_value = x[feature_id]

        if isnan(feature_value) or feature_value is None or (tree[MISSING_TYPE] == "Zero" and feature_value == 0):
            go_left = tree[DEFAULT_LEFT]
            sign = "?"
        elif tree[DECISION_TYPE] == "<=":
            go_left = feature_value <= threshold
            sign = "<=" if go_left else ">"
        else:  # tree[DECISION_TYPE] == "==":
            go_left = feature_value == int(threshold)
            sign = "==" if go_left else "!="

        path.append(
            (depth, "B" + str(tree[SPLIT_INDEX]), feature_names[feature_id], feature_value, sign, threshold,
             tree[INTERNAL_VALUE], tree[SPLIT_GAIN], tree[INTERNAL_COUNT])
        )
        tree = tree[LEFT_CHILD] if go_left else tree[RIGHT_CHILD]
        depth += 1
    else:
        path.append((depth, "L" + str(tree[LEAF_INDEX]), "", "", "", "", tree[LEAF_VALUE], "", tree[LEAF_COUNT]))

    return TreePath(path, optional_header=["Node Value", "Split Gain", "Sample Number"])
