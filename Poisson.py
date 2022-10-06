from inspect import ArgInfo
import cv2
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from util import *

"""
Poisson Image Editing

input: target image, laplacian image
output: integrated image
"""


# Create poisson matrix A
def create_poisson_matrix(h, w):
    """
    Create poisson matrix A
    A is a sparse matrix, so we use lil_matrix.
    for i in range(h) and j in range(w);
    row A[i*w+j, :] represents frame[i, j] pixel's Poisson equation.
    A[i, i] represents the number of frame[i, j]'s neigbor pixels, and if let n the neigbor pixel of frame[i, j], A[i, n] is -1.
    """

    size = h*w
    A = lil_matrix((size, size))
    n = 0

    for i in range(h):
        for j in range(w):
            idx = i*w + j
            # number of neigbors
            Np = 0

            dw = [0, 1, 0, -1]
            dh = [1, 0, -1, 0]

            # find inside neigbors
            for ww, hh in zip(dw, dh):
                if i+hh >= 0 and i+hh < h and j+ww >= 0 and j+ww < w:
                    Np += 1
                    n += 1
                    A[idx, (i+hh)*w + (j+ww)] = -1

            A[idx, idx] = Np
            n += 1

    print( n / size )
    A = A.tocsr()
    return A

# Create poisson b vector
def create_poission_b(laplacian) -> np.ndarray:
    return laplacian.flatten()

# Poisson Image Editing
def process(target, laplacian, A) -> np.ndarray:
    h, w = target.shape[:2]
    # Create poisson A matrix. Ax = b.
    # print('Create a poisson matrix A...')
    # A = create_poisson_matrix(h, w)

    # Create b vector
    print('Create a poisson b vector...')
    b = create_poission_b(laplacian)

    # create an initial answer of x
    x0 = target.flatten()

    # Solve Ax = b
    print('Solve Ax = b...')
    x = linalg.isolve.bicg(A, b, maxiter=8, x0=x0)

    # x = linalg.dsolve.spsolve(A, b)
    y = x[0] / x[0].max() * 255
    # y = x / x.max() * 255
    print(y)
    y = y.astype(np.uint8)
    # print(x.shape)

    composite = np.zeros_like(target)
    for i in range(h):
        for j in range(w):
            composite[i, j] = y[i*w + j]
    
    return composite
