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
    else:
        f = np.fft.fft2(nimg)
        fft_mat = 20*np.log(np.abs(f))

    # Transposition, the low frequency part moves to the middle, the high frequency part moves to the surrounding
    if is_shift:
        return np.fft.fftshift(fft_mat)
    else:
        return fft_mat


def to_fft_image(fft_mat):
    """Convert frequency matrix to visual image"""

    # Add 1 to the log function to avoid log (0).
    log_mat = cv2.log(1 + cv2.magnitude(fft_mat[:, :, 0], fft_mat[:, :, 1]))

    # Standardized to between 0 ~ 255
    cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(log_mat))


def ifft(fft_mat, use_cv2=True):
    """Inverse Fourier transform, return inverse transform image"""
    # Reverse transposition, the low frequency part moves to the surrounding, the high frequency part moves to the middle
    f_ishift_mat = np.fft.ifftshift(fft_mat)

    if use_cv2:
        # Inverse Fourier Transform
        img_back = cv2.idft(f_ishift_mat)

        # Convert complex number to amplitude, sqrt (re ^ 2 + im ^ 2)
        img_back = cv2.magnitude(*cv2.split(img_back))

        # Standardized to between 0 ~ 255
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(np.around(img_back))
        return img_back
    else:
        f_ishift_mat = np.fft.ifftshift(fft_mat)
        dst_ifft = np.fft.ifft2(f_ishift_mat)
        img_back = np.real(dst_ifft)
        img_back = np.uint8(np.clip(img_back, 0, 255))
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
