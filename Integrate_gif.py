import cv2
import numpy as np
from util import *
import Poisson
import argparse
from tqdm import tqdm
import time

"""
Retrive interior movie from windsheild movie by integrating laplacian of interior movie

input: windsheild image, human's laplacian image
output: interiror image
"""

def main(target_path, laplacian_path, output_path, color, iter):
    # read movie
    # target_movie = cv2.VideoCapture(target_path)
    if color != 1:
        target = cv2.imread(target_path, 0)
        laplacian = cv2.imread(laplacian_path, 0)
    else:
        target = cv2.imread(target_path)
        laplacian = cv2.imread(laplacian_path)

    h, w = target.shape[:2]


    # get frame size
    # w = int(target_movie.get(3))
    # h = int(target_movie.get(4))

    # get frame rate
    # frame_rate = target_movie.get(cv2.CAP_PROP_FPS)

    # get total frame
    # total_frame = int(target_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # set output video
    # writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))

    # for _ in tqdm(range(total_frame)):
        # get frame
        # target_frame = target_movie.read()[1]
        # cv2.imshow('img', target_frame)
        # cv2.waitKey(1)
        # target_frame = cv2.resize(target_frame, (h, w))


    # Poisson Image Editing
    cv2.imshow('img', target)
    cv2.waitKey(1)

    # target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    # laplacian = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)

    print(color)
    print(type(color))
    print(color!=1)

    print('Create a poisson matrix A...')
    A = Poisson.create_poisson_matrix(h, w)

    imgs = []

    for i in range(iter[0], iter[1], iter[2]):
        if color!=1:
            result = Poisson.process(target, laplacian, A, i)
        else:
            integrated_bgr = [Poisson.process(target[:, :, i], laplacian[:, :, i], A, i) for i in range(3)]
            result = cv2.merge(integrated_bgr)
        imgs.append(result)

    # writer.write(result)

    # cv2.imwrite('result.png', result)

    # cv2.imshow('result', result)
    # cv2.waitKey(1)

    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # result_laplacian = cv2.Laplacian(result, cv2.CV_32F)
    # cv2.imshow('laplacian', result_laplacian)
    # cv2.waitKey(0)

    make_gif(imgs, 5, output_path)


    # writer.release()

def main_cv2(target_path, laplacian_path, output_path):
    # read movie
    target_movie = cv2.VideoCapture(target_path)
    laplacian = cv2.imread(laplacian_path)

    # get frame size
    w = int(target_movie.get(3))
    h = int(target_movie.get(4))

    # get frame rate
    frame_rate = target_movie.get(cv2.CAP_PROP_FPS)

    # get total frame
    total_frame = int(target_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # set output video
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))

    mask = np.ones(laplacian.shape, dtype=np.uint8) * 255

    for _ in tqdm(range(total_frame)):
        # get frame
        target_frame = target_movie.read()[1]

        # Poisson Image Editing
        result = cv2.seamlessClone(laplacian, target_frame, mask, (h//2, w//2), cv2.NORMAL_CLONE)

        cv2.imshow('result', result)
        cv2.waitKey(1)

        writer.write(result)

    writer.release()


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image', required=True, type=str, help='path to windsheild image')
    arg.add_argument('-l', '--laplacian', required=True, type=str, help='path to laplacian image')
    arg.add_argument('-o', '--output', required=True, type=str, help='path to output movie')
    arg.add_argument('-c', '--color', default=1, type=int, help='0: grayscale 1: color')
    arg.add_argument('-s', '--iter_start', default=1, type=int, help='start iteration')
    arg.add_argument('-e', '--iter_end', default=1, type=int, help='end iteration')
    arg.add_argument('-t', '--iter_step', default=1, type=int, help='step iteration')

    args = vars(arg.parse_args())
    iter = [args['iter_start'], args['iter_end'], args['iter_step']]
    
    main(args['image'], args['laplacian'], args['output'], args['color'], iter)
    # main_cv2(args['movie'], args['laplacian'], args['output'])
