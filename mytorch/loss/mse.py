from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    error = preds - actual      
    squared_error = error ** 2    
    return squared_error.sum() / Tensor(preds.data.size)  
