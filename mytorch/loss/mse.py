from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    diff = (preds - actual)
    sqDiff = diff ** 2
    sum = sqDiff.sum()
    num = preds.data.size
    return sum * (1 /num)
