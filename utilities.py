import numpy as np

LIST_LIKE = "list_like"
DICT_LIKE = "dict_like"


def import_decorator(func):
    """Decorate a function to handle import error.

    Args:
        func: Function to be decorated.

    Returns:
        A decorated function.
    """
    def decorated_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError:
            return None
    return decorated_func


def format_input(x, format_type):
    """Format the input feature vector.

    Args:
        x: A feature vector.
        format_type (str): Either list_like or dict_like.

    Returns:
        Data in the desired format.

    Raises:
        TypeError: If x is neither a list, a numpy.ndarray, nor a pandas.Series.
    """
    for f in [_format_list_numpy, _format_pandas, _format_dmatrix]:
        result = f(x, format_type=format_type)
        if result is not None:
            return result

    raise TypeError("x should be a list, a numpy.ndarray or a pandas.Series.")


def _format_list_numpy(x, format_type):
    if isinstance(x, (list, np.ndarray)):
        if isinstance(x, np.ndarray):
            x = x.flatten()

        if format_type == LIST_LIKE:
            return x, ["Feature_{}".format(i) for i in range(len(x))]
        elif format_type == DICT_LIKE:
            return {"f{}".format(i): x[i] for i in range(len(x))}


@import_decorator
def _format_pandas(x, format_type):
    import pandas as pd
    if isinstance(x, pd.Series):
        if format_type == LIST_LIKE:
            return x.values.flatten(), x.index.tolist()
        elif format_type == DICT_LIKE:
            x.index = x.index.str.replace(" ", "")
            return x


# noinspection PyUnusedLocal
@import_decorator
def _format_dmatrix(x, format_type):
    from xgboost import DMatrix
    if isinstance(x, DMatrix):
        raise TypeError("DMatrix is currently not supported.")
