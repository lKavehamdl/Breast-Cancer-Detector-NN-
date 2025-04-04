import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    x_exp = x.exp()
    neg_x_exp = (-x).exp()
    return (x_exp - neg_x_exp) * ((x_exp + neg_x_exp) ** -1)