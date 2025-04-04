from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        # x.shape[0] is number of samples
        x = np.pad(x, ((0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0)))
        n1 = x.shape[1] - self.kernel_size[0]
        n2 = x.shape[2] - self.kernel_size[1]
        y = Tensor(data=np.zeros((x.shape[0], n1//self.stride[0] + 1, n2//self.stride[1] + 1, self.out_channels)))
        for t in range(x.shape[0]):
            for i in range(0, n1, self.stride[0]):
                for j in range(0, n2, self.stride[1]):
                    for k in range(self.out_channels):
                        y[t, i//self.stride[0], j//self.stride[1], k] = x[t, i:i+self.kernel_size[0], j:j+self.kernel_size[1], :] * self.weight + self.bias

        return y
    
    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer((*self.kernel_size, self.in_channels, self.out_channels), 
                             self.kernel_size[0] * self.kernel_size[1] * self.in_channels,
                             self.kernel_size[0] * self.kernel_size[1] * self.out_channels,
                             self.initialize_mode),
            requires_grad=True
        )

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, 1, 1, self.out_channels),
                                 self.kernel_size[0] * self.kernel_size[1] * self.in_channels,
                                 self.kernel_size[0] * self.kernel_size[1] * self.out_channels,
                                 self.initialize_mode),
                requires_grad=True
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()

        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return np.stack((self.weight.data, np.broadcast_to(self.bias.data, (*self.kernel_size, self.in_channels, self.out_channels))))
        else:
            return self.weight.data
        
    def grad(self):
        if self.need_bias:
            return np.stack((self.weight.grad.data, np.broadcast_to(self.bias.grad.data, (*self.kernel_size, self.in_channels, self.out_channels))))
        else:
            return self.weight.grad.data
        
    def update_parameters(self, weight):
        if self.need_bias:
            self.weight.data = weight[0, :, :, :, :]
            self.bias.data = weight[1, 0, 0, 0, :].reshape((1, 1, 1, -1))
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
