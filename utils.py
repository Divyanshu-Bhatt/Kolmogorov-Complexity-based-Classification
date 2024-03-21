import numpy as np
from binary_fractions import Binary
import decimal


def sizeCalculator(value):
    """
    Calculate the size of the value in bits

    Parameters
    ----------
    value : float or tuple or list or int
        The value to calculate the size of
    """

    if isinstance(value, (int, float)):
        return len(str(Binary(value))[2:])
    elif isinstance(value, (tuple, list, np.ndarray)):
        size = 0
        for val in value:
            size += sizeCalculator(val)
        return size
    elif value is None:
        return 0
    elif isinstance(value, (decimal.Decimal)):
        if value == 0:
            return 0
        return sizeCalculator(int(str(value).split(".")[1]))
    elif isinstance(value, str):
        return len(value)
