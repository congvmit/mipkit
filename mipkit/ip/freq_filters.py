"""
 The MIT License (MIT)
 Copyright (c) 2021 Cong Vo
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 Provided license texts might have their own copyrights and restrictions
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
"""

from .fft import fft_distances
import numpy as np
import cv2


def get_H(m, n):
    """Calculate the distance of each point of the m, n matrix from the center"""
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v.shape = n, 1
    return (u - m/2)**2 + (v - n/2)**2


def laplacian_filter(fft_mat, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    """Homomorphic Filter

    Args:
        fft_mat (ndarray): fft matrix
        d0 (int, optional): filter size. Defaults to 10.
        r1 (float, optional): [description]. Defaults to 0.5.
        rh (int, optional): [description]. Defaults to 2.
        c (int, optional): [description]. Defaults to 4.
        h (float, optional): [description]. Defaults to 2.0.
        l (float, optional): [description]. Defaults to 0.5.

    Returns:
        nd]: [description]
    """
    rows, cols = fft_mat.shape[:2]

    Huv = np.zeros_like(fft_mat)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2),
                       np.arange(-rows//2, rows//2))
    # Huv = -((M - M//2) ** 2 + (N - N//2) ** 2)
    Huv = get_H(rows, cols)

    print(Huv)
    # Gaussian high-pass filter
    # Z = (rh - r1) * (1 - np.exp(-c * (Duv ** 2 / d0 ** 2))) + r1

    # Z =  (1 - np.exp(-c * (Duv ** 2 / d0 ** 2)))
    dst = Huv * fft_mat

    # Huv = (h - l) * Huv + l
    return dst


def homomorphic_filter(fft_mat, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    """Homomorphic Filter

    Args:
        fft_mat (ndarray): fft matrix
        d0 (int, optional): filter size. Defaults to 10.
        r1 (float, optional): [description]. Defaults to 0.5.
        rh (int, optional): [description]. Defaults to 2.
        c (int, optional): [description]. Defaults to 4.
        h (float, optional): [description]. Defaults to 2.0.
        l (float, optional): [description]. Defaults to 0.5.

    Returns:
        nd]: [description]
    """
    rows, cols = fft_mat.shape[:2]

    Huv = np.zeros_like(fft_mat)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2),
                       np.arange(-rows//2, rows//2))
    Duv = np.sqrt(M ** 2 + N ** 2)

    # Gaussian high-pass filter
    Z = (rh - r1) * (1 - np.exp(-c * (Duv ** 2 / d0 ** 2))) + r1

    # Z =  (1 - np.exp(-c * (Duv ** 2 / d0 ** 2)))
    Huv = Z * fft_mat

    Huv = (h - l) * Huv + l
    return Huv


def lpfilter(fft_mat, flag: int, d0: int, n: int, rows=None, cols=None):
    """Low-pass filter

    Args:
        fft_mat (ndarray): fft matrix
        flag (int): filter type
            0-ideal low-pass filtering
            1-Butterworth low-pass filtering
            2-Gaussian low-pass filtering
        rows (int): the height of the filtered matrix
        cols (int): the width of the filtered matrix
        d0 (int): filter size d0
        n (int): order of Butterworth low-pass filtering

    Returns:
        ndarray: filter matrix
    """
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None

    if rows is None and cols is None:
        rows, cols = fft_mat.shape[:2]
    else:
        assert rows is not None and cols is not None

    # Ideal low-pass filtering
    if flag == 0:
        filter_mat = np.zeros((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 2, cols // 2),
                   d0, (1, 1, 1), thickness=-1)

    # Butterworth low-pass filtering
    elif flag == 1:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
        # fft_mat has 2 channels, real and imaginary
        # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))

    # Gaussian low-pass filtering
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv * duv) / (2 * d0 * d0))
        # fft_mat has 2 channels, real and imaginary
        # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))

    filtered_mat = filter_mat * fft_mat
    return filtered_mat


def hpfilter(fft_mat, flag, d0, n, rows=None, cols=None):
    """High-pass filter

    Args:
        fft_mat (ndarray): fft matrix   
        flag (int): filter type
            0-ideal high-pass filtering
            1-Butterworth high-pass filtering
            2-Gaussian high-pass filtering
        rows (int): the height of the filtered matrix
        cols (int): the width of the filtered matrix
        d0 (int): filter size d0
        n (int): the order of Butterworth high-pass filtering

    Returns:
        ndarray: filter matrix b
    """
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None

    if rows is None and cols is None:
        rows, cols = fft_mat.shape[:2]
    else:
        assert rows is not None and cols is not None

    # Ideal high-pass filtering
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 2, cols // 2),
                   d0, (0, 0, 0), thickness=-1)

    # Butterworth high-pass filtering
    elif flag == 1:
        duv = fft_distances(rows, cols)
        # duv has a value of 0 (the center is 0 from the center). To avoid division by 0, set the center to 0.000001
        duv[rows // 2, cols // 2] = 0.000001
        filter_mat = 1 / (1 + np.power(d0 / duv, 2 * n))
        # fft_mat has 2 channels, real and imaginary
        # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))

    # Gaussian high-pass filtering
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 - np.exp(-(duv * duv) / (2 * d0 * d0))
        # fft_mat has 2 channels, real and imaginary
        # fliter_mat also requires 2 channels
        filter_mat = cv2.merge((filter_mat, filter_mat))

    filtered_mat = filter_mat * fft_mat
    return filtered_mat


def combine_images(images: list, axis=1):
    """Combine images

    Args:
        images (list): image list (must have the same dimension)
        axis (int): merge direction
            When axis = 0, the images are merged vertically;
            When axis = 1, the images are merged horizontally.
    Returns:
        merge image
    """
    ndim = images[0].ndim
    shapes = np.array([mat.shape for mat in images])
    assert np.all(map(lambda e: len(e) == ndim, shapes)
                  ), 'all images should be same ndim.'
    if axis == 0:  # merge images vertically

        # Merge image cols
        cols = np.max(shapes[:, 1])

        # Expand the cols size of each image to make the cols consistent
        copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # Merge vertically
        return np.vstack(copy_imgs)

    else:  # merge images horizontally
        # Combine the rows of the image
        rows = np.max(shapes[:, 0])

        # Expand the row size of each image to make rows consistent
        copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]

        # Merge horizontally
        return np.hstack(copy_imgs)
