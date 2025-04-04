from mytorch import Tensor, Dependency

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    num = preds.data.size
    return -1 * (label * preds.log() + ((Tensor(1.0) - label) * (Tensor(1.0) - preds).log())).sum() * (Tensor(1 / num))

