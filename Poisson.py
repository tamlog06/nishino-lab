import cv2
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from util import *
from tqdm import tqdm

"""
Poisson Image Editing

input: target image, laplacian image
output: integrated image
"""


# Create poisson matrix A
def create_poisson_matrix(h, w):
    """
    Create poisson matrix A.
    Each row A satisfies the below matrix:
        sum(fq) - Np*fp  = gp, where 
            Np: number of fp's neigbor indexes
            fp: image's p index
            fq: image's p index's neigbor indexes
            gp: laplacian' p index
        Af = g
    A is a sparse matrix, so we use lil_matrix.
    for i in range(h) and j in range(w);
    row A[i*w+j, :] represents image[i, j] pixel's Poisson equation.
    A[i, i] represents the number of image[i, j]'s neigbor pixels, and if let n the neigbor pixel of frame[i, j], A[i, n] is -1.
    """

    size = h*w
    A = lil_matrix((size, size))
    n = 0

    for i in tqdm(range(h)):
        for j in range(w):
            idx = i*w + j
            # number of neigbors
            Np = 0

            dw = [0, 1, 0, -1]
            dh = [1, 0, -1, 0]

            # find inside neigbors
            for ww, hh in zip(dw, dh):
                if i+hh >= 0 and i+hh < h and j+ww >= 0 and j+ww < w:
                    # Np += 1
                    Np -= 1
                    n += 1
                    # A[idx, (i+hh)*w + (j+ww)] = -1
                    A[idx, (i+hh)*w + (j+ww)] = 1

            A[idx, idx] = Np
            n += 1

    print( n / size )
    A = A.tocsr()
    return A




# Poisson Image Editing
def process(target, laplacian, A, maxiter, solver) -> np.ndarray:
    """
    input:
        target: target image
        laplacian: laplacian image
        A: Poisson lil_matrix
        maxiter: max iteration of iterative solution
        solver: the way of iterative solution; choose one of the below
            ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'minres', 'qmr']
    output:
        composite: integrated image
    """
    h, w = target.shape[:2]
    # Create poisson A matrix. Ax = b.
    # print('Create a poisson matrix A...')
    # A = create_poisson_matrix(h, w)

    # Create b vector
    b = laplacian.flatten()

    # create an initial answer of x
    x0 = target.flatten()

    # Solve Ax = b
    assert solver in ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'minres', 'qmr']
    x = eval(f'linalg.isolve.{solver}')(A, b, maxiter=maxiter, x0=x0)[0]

    # print(x.max(), x.min())
    if x.min() < 0 :
        x = x - x.min()
    x = x / x.max() * 255
    x = np.uint8(x)

    composite = np.reshape(x, (h, w))
    
    return composite
