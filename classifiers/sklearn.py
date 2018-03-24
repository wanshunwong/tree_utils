from tree_path import TreePath
from utilities import format_input, LIST_LIKE

# indices of different attributes in tree_.__getstate__()["nodes"]
LEFT_CHILD = 0
RIGHT_CHILD = 1
FEATURE = 2
THRESHOLD = 3
IMPURITY = 4
N_NODES_SAMPLES = 5
WEIGHTED_N_NODES_SAMPLES = 6


def prediction_path_sklearn(tree, x):
    """Compute the decision path in a sklearn tree, and return a TreePath object.

    Args:
        tree: A sklearn tree.
        x: A feature vector

    Returns:
        A TreePath object.
    """
    x, feature_names = format_input(x, LIST_LIKE)
    path = []
    nodes = tree.tree_.__getstate__()["nodes"]
    node_id = 0
    node = nodes[node_id]
    depth = 0

    while node[LEFT_CHILD] != -1:  # not a leaf node
        feature_id = node[FEATURE]
        threshold = node[THRESHOLD]
        feature_value = x[feature_id]
        sign_bool = feature_value <= threshold
        sign = "<=" if sign_bool else ">"
        path.append(
            (depth, node_id, feature_names[feature_id], feature_value, sign, threshold,
             node[IMPURITY], node[N_NODES_SAMPLES], node[WEIGHTED_N_NODES_SAMPLES])
        )

        node_id = node[LEFT_CHILD] if sign_bool else node[RIGHT_CHILD]
        node = nodes[node_id]
        depth += 1
    else:
        path.append(
            (len(path), node_id, "", "", "", "",
             node[IMPURITY], node[N_NODES_SAMPLES], node[WEIGHTED_N_NODES_SAMPLES])
        )
    return TreePath(path, optional_header=["Impurity", "Sample Number", "Sample Weight"])
