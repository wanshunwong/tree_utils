from tabulate import tabulate


class TreePath:
    """An object describing a decision path.

    Attributes:
        path (list): A list of tuples containing the data of all the nodes in the decision path.
        optional_header (list): A list of strings describing the optional columns.
    """

    def __init__(self, path, optional_header):
        self.path = path
        self.header = ["Depth", "Node", "Feature", "Feature Value", "Sign", "Threshold"] + optional_header

    def __repr__(self):
        return repr(self.path)

    def __str__(self):
        return tabulate(
            self.path,
            headers=self.header,
            tablefmt="fancy_grid"
        )

    def node_id(self, depth):
        """Return the id of the node at a particular depth.

        Args:
            depth (int): Depth of the node in the decision path.

        Returns:
            int: Id of a node.
        """
        return self.path[depth][1]

    def feature(self, depth):
        """Return the name of the feature of the node at a particular depth.

        Args:
            depth (int): Depth of the node in the decision path.

        Returns:
            str: Name of a feature.
        """
        return self.path[depth][2]

    def value(self, depth):
        """Return the feature value of the node at a particular depth.

        Args:
            depth (int): Depth of the node in the decision path.

        Returns:
            float: Value of a feature.
        """
        return self.path[depth][3]

    def threshold(self, depth):
        """Return the feature threshold of the node at a particular depth.

        Args:
            depth (int): Depth of the node in the decision path.

        Returns:
            float: Threshold of a feature.
        """
        return self.path[depth][5]

    def compare(self, tree_path):
        """Compare itself with another TreePath object.

        Args:
            tree_path (TreePath): A TreePath object.

        Returns:
            None.
        """
        depth = 0
        common_nodes = []
        while depth < len(self.path):
            current_node_id = self.node_id(depth)
            if current_node_id == tree_path.node_id(depth):
                common_nodes.append(str(current_node_id))
                depth += 1
            else:
                break

        print("The common nodes are {}.".format(", ".join(common_nodes)))
        if depth < len(self.path):
            pre_depth = depth - 1
            print("At depth {}, node {}, for feature {},".format(pre_depth, common_nodes[-1], self.feature(pre_depth)))
            print("the values are {} and {}, the threshold is {}".format(
                self.value(pre_depth), tree_path.value(pre_depth), self.threshold(pre_depth)
            ))
        else:
            print("The two paths are identical.")
