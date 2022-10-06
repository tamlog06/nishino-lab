import cv2
import numpy as np 


if __name__ == '__main__':
    video = cv2.VideoCapture('homography.mp4')
    while True:
        ret, frame = video.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    video.release()
