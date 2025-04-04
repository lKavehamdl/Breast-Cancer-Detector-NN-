import numpy as np
from mytorch import Tensor, Dependency

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    diff = preds - actual
    sqDiff = diff.__pow__(2)
    sum = sqDiff.sum()
    num = Tensor(np.prod(sqDiff.shape))
    mse = sum.div(num)
    if preds.requires_grad or actual.requires_grad:
        mse.requires_grad = True
        
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Gradient of MSE: 2*(preds-actual)/n
            n = np.prod(preds.shape)
            return (2 * (preds.data - actual.data) / n) * grad
            
        mse.depends_on = [
            Dependency(preds, grad_fn),
            Dependency(actual, lambda g: -grad_fn(g))  # Actual gets negative gradient
        ]
        
    return mse