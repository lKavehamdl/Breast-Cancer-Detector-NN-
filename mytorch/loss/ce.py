from mytorch import Tensor
import math

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    eps = Tensor(1e-9)  # Small epsilon to prevent log(0)
    log_probs = (preds + eps).log()  #
    loss = -(label * log_probs).sum() / Tensor(preds.data.shape[0])  
    return loss