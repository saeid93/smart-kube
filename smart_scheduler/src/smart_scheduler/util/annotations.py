import numpy as np
from .constants import PRECISION

def override(cls):
    """Annotation for documenting method overrides.
    Args:
        cls (type): The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.
    """

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(
                method, cls))
        return method

    return check_override

def rounding(func):
    """
    rounding the variable to the approperiate precession
    """
    def round_the_result(*args, **kwargs):
        return np.round(func(*args, **kwargs), PRECISION)
    return round_the_result