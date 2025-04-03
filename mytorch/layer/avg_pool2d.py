from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, channels, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        # Calculate output dimensions
        out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
        
        # Pad the input
        padded_x = np.pad(x.data, 
                         ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                         mode='constant')
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_h, out_w))
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * stride_h
                        w_start = w * stride_w
                        receptive_field = padded_x[b, c, h_start:h_start+k_h, w_start:w_start+k_w]
                        output[b, c, h, w] = np.mean(receptive_field)
        
        return Tensor(output, requires_grad=x.requires_grad) 
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
