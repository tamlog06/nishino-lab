import numpy as np
import cv2

# return homograpy transformed image
def homography_transform(img, points):
    h, w = img.shape[:2]

    # coordinates are witten in (w, h) format
    target_points = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
    size = (w, h)

    mat = cv2.getPerspectiveTransform(points, target_points)
    perspective_image = cv2.warpPerspective(img, mat, size)

    return perspective_image
