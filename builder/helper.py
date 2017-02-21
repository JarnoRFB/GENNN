import numpy as np
import math

def get_tensor_size(tensor):
    """Calculate the number of elements in tensor, assuming first dim is None."""
    return int(np.prod(tensor.get_shape()[1:]))
