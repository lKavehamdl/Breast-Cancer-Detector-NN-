import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor) -> Tensor:
    """
    TODO: implement leaky_relu function.
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """

    neg_slope = 0.01
    data = np.maximum(x.data, np.zeros_like(x.data)) + neg_slope * np.minimum(x.data, np.zeros_like(x.data))
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # return np.where((data > 0), grad, np.zeros_like(data)) + \
            #        np.where((data < 0), neg_slope * grad, np.zeros_like(data))
            return grad * np.where(x.data > 0, 1, neg_slope)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
