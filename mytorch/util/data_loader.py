import numpy as np
from mytorch import Tensor
from PIL import Image
from random import shuffle
from typing import List, Tuple

class DataLoader:
    def __init__(self, x_train, y_train, batch_size, shuffle=True):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.x_train):
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indexes)
            raise StopIteration
        if self.current_index == 0:
            np.random.shuffle(self.indexes)
        
        batch_indexes = self.indexes[self.current_index:self.current_index + self.batch_size]
        batch_x = self.x_train[batch_indexes]
        batch_y = self.y_train[batch_indexes]
        self.current_index += self.batch_size
        return batch_x, batch_y
    
    def __len__(self):
        return (len(self.x_train) + self.batch_size - 1) // self.batch_size  # Ceiling division
