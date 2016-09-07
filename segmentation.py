import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def create_image(size):
    img = np.zeros([size, size])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = i + j
    return img


def magic_wand(x, sx, sy, t):
    y = np.zeros(x.shape)
    xs = x[sx, sy]
    thres = [xs-t, xs+t]
    flood(x, y, sx, sy, thres)
    return y


def flood(x, y, sx, sy, thres):
    if thres[0] < x[sx, sy] < thres[1]:
        y[sx, sy] = 1
        if sx-1 >= 0:
            flood(x, y, sx-1, sy, thres)
        if sy-1 >= 0:
            flood(x, y, sx, sy-1, thres)
        if sx+1 <= x.shape[0]:
            flood(x, y, sx+1, sy, thres)
        if sy+1 <= x.shape[1]:
            flood(x, y, sx, sy+1, thres)


if __name__ == '__main__':
    directory = os.path.dirname(os.path.realpath(__file__))
    # img = mpimg.imread(directory + '\\images\\butterfly.pgm')

    x = create_image(255)
    plt.imshow(x, cmap='Greys_r')
    plt.show()

    y = magic_wand(x, 100, 100, 10)
    plt.imshow(y, cmap='Greys_r')
    plt.show()