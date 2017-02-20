import numpy as np
import math

def get_tensor_size(tensor):
    """Calculate the number of elements in tensor, assuming first dim is None."""
    return int(np.prod(tensor.get_shape()[1:]))

def is_prime(n):
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))
