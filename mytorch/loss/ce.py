from mytorch import Tensor, Dependency
import numpy as np

def CategoricalCrossEntropy(preds: Tensor, label: Tensor, batch_size: int):
    "TODO: implement Categorical Cross Entropy loss"
    eps = Tensor(1e-9)
    preds_clipped = np.clip(preds.data, eps, 1 - eps)  # Avoid log(0)
    data = -np.mean(np.sum(label.data * np.log(preds_clipped), axis=1))  # Compute CE loss
    requires_grad = preds.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
           
            return grad * (-label.data / preds_clipped) / preds.data.shape[0]  # Normalize over batch

        depends_on = [Dependency(preds, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)