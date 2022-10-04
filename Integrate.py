import cv2
import numpy as np
from util import *
import Poisson
import argparse
from tqdm import tqdm

"""
Retrive interior movie from windsheild movie by integrating laplacian of interior movie

input: windsheild image, human's laplacian image
output: interiror image
"""

def main(target_path, laplacian_path, output_path):
    # read movie
    target_movie = cv2.VideoCapture(target_path)
    laplacian = cv2.imread(laplacian_path)

    # get frame size
    w = int(target_movie.get(3)) // 10
    h = int(target_movie.get(4)) // 10

    print(w, h)

    laplacian = cv2.resize(laplacian, (w, h))

    # get frame rate
    frame_rate = target_movie.get(cv2.CAP_PROP_FPS)

    # get total frame
    total_frame = int(target_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # set output video
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))

    for _ in tqdm(range(total_frame)):
        # get frame
        target_frame = target_movie.read()[1]
        target_frame = cv2.resize(target_frame, (h, w))

        # Poisson Image Editing
        integrated_bgr = [Poisson.process(target_frame[:, :, i], laplacian[:, :, i]) for i in range(3)]
        result = cv2.merge(integrated_bgr)

        cv2.imshow('result', result)
        cv2.waitKey(1)

        writer.write(result)

    writer.release()

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
    arg.add_argument('-m', '--movie', required=True, help='path to windsheild movie')
    arg.add_argument('-l', '--laplacian', required=True, help='path to laplacian image')
    arg.add_argument('-o', '--output', required=True, help='path to output movie')

    args = vars(arg.parse_args())
    
    main(args['movie'], args['laplacian'], args['output'])
    # main_cv2(args['movie'], args['laplacian'], args['output'])
