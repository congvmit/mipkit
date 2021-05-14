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

import numpy as np
import cv2


def fft(img: np.ndarray, is_shift=True, use_cv2=True):
    """Fourier transform the image and return the frequency matrix after transposition"""
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]

    # Calculate the optimal size
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)

    # According to the new size, create a new transformed image
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img

    # Fourier transform
    if use_cv2:
        fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)

        # Convert to complex matrix
        fft_mat = fft_mat[:, :, 0] + fft_mat[:, :, 1] * 1j
    else:
        fft_mat = np.fft.fft2(nimg)

    # Transposition, the low frequency part moves to the middle, the high frequency part moves to the surrounding
    if is_shift:
        return np.fft.fftshift(fft_mat), rows, cols
    else:
        return fft_mat, rows, cols


def to_fft_image(fft_mat, is_shift=False):
    """Convert frequency matrix to visual image"""
    if is_shift:
        fft_mat = np.fft.fftshift(fft_mat)
    log_mat = 20*np.log(np.abs(fft_mat))
    return np.uint8(np.around(log_mat))


def ifft(fft_mat, rows, cols):
    """Inverse Fourier transform, return inverse transform image

    Args:
        fft_mat (ndarray]): fft matrix
        rows (int): the height of the original image
        cols (int): the width of the original image 

    Returns:
        ndarray: inversed image
    """

    # Reverse transposition, the low frequency part moves to the surrounding, the high frequency part moves to the middle
    f_ishift_mat = np.fft.ifftshift(fft_mat)

    dst_ifft = np.fft.ifft2(f_ishift_mat)
    img_back = np.real(dst_ifft)
    img_back = np.uint8(np.clip(img_back, 0, 255))
    img_back = img_back[:rows, :cols]
    return img_back


def fft_distances(m, n):
    """Calculate the distance of each point of the m, n matrix from the center"""
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v.shape = n, 1

    # The distance from each point to the upper left corner of the matrix
    ret = np.sqrt(u * u + v * v)

    # The distance of each point from the center of the matrix
    return np.fft.fftshift(ret)
