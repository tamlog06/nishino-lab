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

def gaussSeidel(A, b, x0, tol, maxiter=1e5):
    xold = x0
    error = 1e12
    
    L = np.tril(A)
    U = A - L
    LInv = np.linalg.inv(L)

    n = 0
    while error > tol and n < maxiter:
        x = np.dot(LInv, b-np.dot(U, xold))
        error = np.linalg.norm(x-xold)/np.linalg.norm(x)
        xold = x
        print(error)
        n += 1

    return x

from PIL import Image
def make_gif(im_list, fps, path):
    duration_time = int(1000.0 / fps)
    print("duration:{}".format(duration_time))
    im_list[0].save(path, save_all=True, optimize=False, append_images=im_list[1:], duration=duration_time, loop=0)

def cv2pil(img):
    new_image = img.copy()
    if new_image.ndim == 2:
        c = 'P'
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        c = 'RGB'
    elif new_image.shape[2] == 4:
        new_image= cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        c = 'RGBA'
    new_image = Image.fromarray(new_image).convert(c)
    return new_image

