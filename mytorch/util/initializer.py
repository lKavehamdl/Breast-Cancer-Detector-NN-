import numpy as np

def xavier_initializer(shape):
    "TODO: implement xavier_initializer" 
    fan_in, fan_out = shape[0], shape[1]
    if len(shape) > 2:
        receptive_field_size = np.prod(shape[2:])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size
    
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float64)

def he_initializer(shape):
    "TODO: implement he_initializer" 
    fan_in = shape[0]
    if len(shape) > 2:
        receptive_field_size = np.prod(shape[2:])
        fan_in *= receptive_field_size
    
    stddev = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, stddev, size=shape).astype(np.float64)


def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    "TODO: implement random_normal_initializer" 
    return np.random.normal(mean, stddev, size=shape).astype(np.float64)

def zero_initializer(shape):
    "TODO: implement zero_initializer" 
    return np.zeros(shape, dtype=np.float64)

def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)

def initializer(shape, mode="random_normal"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
