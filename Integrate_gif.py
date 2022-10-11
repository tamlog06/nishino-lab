import cv2
import numpy as np
from util import *
import Poisson
import argparse
from tqdm import tqdm
from PIL import Image

"""
Retrive interior movie from windsheild movie by integrating laplacian of interior movie

input: windsheild image, human's laplacian image
output: interiror image
"""

def main(target_path, gt_path, output_name, gray, iter):
    # read image
    if gray:
        print('gray mode')
        # read as gray scale
        target = cv2.imread(target_path, 0)
        laplacian = cv2.imread(gt_path, 0)
        laplacian = cv2.Laplacian(laplacian, cv2.CV_64F)
    else:
        target = cv2.imread(target_path)
        laplacian = cv2.imread(gt_path)
        b, g, r = laplacian[:, :, 0], laplacian[:, :, 1], laplacian[:, :, 2]
        b_laplacian = cv2.Laplacian(b, cv2.CV_64F)
        g_laplacian = cv2.Laplacian(g, cv2.CV_64F)
        r_laplacian = cv2.Laplacian(r, cv2.CV_64F)
        laplacian = cv2.merge([b_laplacian, g_laplacian, r_laplacian])

    h, w = target.shape[:2]

    print(laplacian.max(), laplacian.min())

    # cv2.imshow('laplacian', laplacian)
    # cv2.waitKey(1)

    print('Create a poisson matrix A...')
    A = Poisson.create_poisson_matrix(h, w)

    # solve Poisson equation by each method below
    solvers = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'minres', 'qmr']
    for solver in solvers:
        print(f"Solve Poisson matrix by {solver}")
        imgs = []
        imgs_laplacian_original = []
        imgs_laplacian = []

        # i specify the max iteration
        for i in tqdm(range(iter[0], iter[1], iter[2])):
            if gray:
                result = Poisson.process(target, laplacian, A, i, solver)
            else:
                integrated_bgr = [Poisson.process(target[:, :, i], laplacian[:, :, i], A, i, solver) for i in range(3)]
                result = cv2.merge(integrated_bgr)

            result_laplacian_original = cv2.Laplacian(result, cv2.CV_64F)
            result_laplacian_original = np.uint8(np.abs(result_laplacian_original))
            result_laplacian = result_laplacian_original/result_laplacian_original.max() *255
            result_laplacian = result_laplacian.astype(np.uint8)

            result = cv2pil(result)
            result_laplacian_original = cv2pil(result_laplacian_original)
            result_laplacian = cv2pil(result_laplacian)

            imgs.append(result)
            imgs_laplacian_original.append(result_laplacian_original)
            imgs_laplacian.append(result_laplacian)

        output_path = f'{output_name}_{solver}.gif'
        output_path_laplacian_original = f'{output_name}_{solver}_laplacian_original.gif'
        output_path_laplacian = f'{output_name}_{solver}_laplacian.gif'

        make_gif(imgs, 2, output_path)
        make_gif(imgs_laplacian_original, 2, output_path_laplacian_original)
        make_gif(imgs_laplacian, 2, output_path_laplacian)

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image', required=True, type=str, help='path to windsheild image')
    arg.add_argument('-l', '--laplacian', required=True, type=str, help='path to GT image')
    arg.add_argument('-o', '--output', required=True, type=str, help='name of output movie')
    arg.add_argument('-g', '--gray', action='store_true', help='if specified, process by  grayscale')
    arg.add_argument('-s', '--iter_start', default=1, type=int, help='start iteration')
    arg.add_argument('-e', '--iter_end', default=1, type=int, help='end iteration')
    arg.add_argument('-t', '--iter_step', default=1, type=int, help='step iteration')

    args = vars(arg.parse_args())
    iter = [args['iter_start'], args['iter_end'], args['iter_step']]
    
    main(args['image'], args['laplacian'], args['output'], args['gray'], iter)
