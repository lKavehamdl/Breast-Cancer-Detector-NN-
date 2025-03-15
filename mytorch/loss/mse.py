import numpy as np
from mytorch import Tensor, Dependency

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    diff = preds - actual
    tmp = np.mean(pow(diff.data, 2))
    req_grad = preds.requires_grad or actual.requires_grad
    depends_on = []
    if req_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (2 / preds.data.size) * diff.data
    
        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data=tmp, requires_grad=req_grad, depends_on=depends_on)