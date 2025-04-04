import numpy as np
from mytorch import Tensor

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    data = x.data.reshape(x.data.shape[0], -1)
    req_grad = x.requires_grad
    depends_on = []
    if req_grad:
        depends_on.append({
            'tensor': x,
            'grad_fn': lambda grad: grad.reshape(x.shape)
        })
    
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
