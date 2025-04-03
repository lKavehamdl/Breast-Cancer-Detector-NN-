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
        batch_size, _, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
        
        padded_x = np.pad(x.data, 
                         ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                         mode='constant')
        
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * stride_h
                        w_start = w * stride_w
                        receptive_field = padded_x[b, :, h_start:h_start+k_h, w_start:w_start+k_w]
                        output[b, c_out, h, w] = np.sum(receptive_field * self.weight.data[c_out])
        
        if self.need_bias:
            output += self.bias.data.reshape(1, -1, 1, 1)
            
        return Tensor(output, requires_grad=x.requires_grad)
    
    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer((self.out_channels, self.in_channels, *self.kernel_size), 
            mode=self.initialize_mode),
            requires_grad=True
        )

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.out_channels,), mode="zero"),
                requires_grad=True
            )


    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        params = [self.weight]
        if self.need_bias:
            params.append(self.bias)
        return params
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
