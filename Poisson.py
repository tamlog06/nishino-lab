import cv2
import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix

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

    for i in range(h):
        for j in range(w):
            idx = i*w + j
            # number of neigbors
            Np = 0

            dx = [0, 1, 0, -1]
            dy = [1, 0, -1, 0]

            # find inside neigbors
            for x, y in zip(dx, dy):
                if i+x >= 0 and i+x < h and j+y >= 0 and j+y < w:
                    Np += 1
                    A[idx, (i+x)*w + (j+y)] = -1

            A[idx, idx] = Np

    return A

# Create poisson b vector
def create_poission_b(laplacian) -> np.ndarray:
    return laplacian.flatten()

# Poisson Image Editing
def process(target, laplacian) -> np.ndarray:
    h, w = target.shape[:2]
    # Create poisson A matrix. Ax = b.
    print('Create poisson matrix A...')
    A = create_poisson_matrix(h, w)

    # Create b vector
    print('Create poisson b vector...')
    b = create_poission_b(laplacian)

    # Solve Ax = b
    print('Solve Ax = b...')
    x = linalg.cg(A, b)

    composite = np.zeros_like(target)
    for i in range(h):
        for j in range(w):
            composite[i, j] = x[0][i*w + j]
    
    return composite
