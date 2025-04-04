from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.01):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for layer in self.layers:
            weight = layer.parameters()
            grad = layer.grad()
            weight = weight - self.learning_rate * grad
            layer.update_parameters(weight)

