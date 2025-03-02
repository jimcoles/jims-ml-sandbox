#  Copyright (c) 2025 James K. Coles (jameskcoles@gmail.com). All rights reserved.

import numpy as np
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray


# Function to multiply two matrices using numpy arrays
def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication on two numpy arrays.

    :param a: First matrix (2D numpy array)
    :param b: Second matrix (2D numpy array)
    :return: Resulting matrix (2D numpy array) after multiplication
    :raises ValueError: If the matrices cannot be multiplied due to incompatible dimensions
    """
    if a.shape[1] != b.shape[0]:
        raise ValueError("Matrix A's columns must match Matrix B's rows for multiplication.")

    prod = np.zeros((a.shape[0], b.shape[1]))
    for idx_a_row in range(a.shape[0]):
        for idx_b_col in range(b.shape[1]):
            prod[idx_a_row][idx_b_col] = dot(a[idx_a_row], b.T[idx_b_col])

    return prod


def dot(a: ndarray, b: ndarray) -> ndarray:

    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("Both arrays must be one-dimensional and of the same length for dot product")

    vector_size = a.shape[0]
    dot_prod = 0
    for idx_vector in range(vector_size):
        dot_prod += a[idx_vector] * b[idx_vector]
    return dot_prod
