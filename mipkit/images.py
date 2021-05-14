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

from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision.transforms.functional as F
import torch
from .utils import deprecated


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


def convert_to_torch_image(img):
    """ Convert the image to a torch image.
        Code:
            img = torch.tensor(img)
            img = img.permute((2, 0, 1)).contiguous()
            if isinstance(img, torch.ByteTensor):
                return img.float().div(255)
            else:
                return img
            return img
    """
    return F.to_tensor(img)


def read_image(file_path, to_rgb=True, vis=False):
    img = cv2.imread(file_path)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if vis:
        show_image(img)
    return img


@deprecated('Only for specific experiments')
def read_image_experiments(file_path: str, to_rgb=True, vis=False, to_tensor=False):
    """Load and convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/master/references/segmentation
    """
    img = read_image(file_path, to_rgb=to_rgb, vis=vis)

    if to_tensor:
        img_tensor = convert_to_torch_image(img)
    return {'img_raw': img,
            'img_tensor': img_tensor
            }


def padding_image(img_arr):
    w, h, c = img_arr.shape
    if w > h:
        padd = (w - h)//2
        img_arr = cv2.copyMakeBorder(
            img_arr.copy(), 10, 10, padd, padd, cv2.BORDER_CONSTANT, value=0)
    elif w < h:
        padd = (h - w)//2
        img_arr = cv2.copyMakeBorder(
            img_arr.copy(), padd, padd, 10, 10, cv2.BORDER_CONSTANT, value=0)
    return img_arr


def _get_type(img):
    if isinstance(img, Image.Image):
        return 'pil'
    elif isinstance(img, np.ndarray):
        return 'np'
    else:
        raise ValueError('Unknown type!')


def _is_ndarray(inp):
    assert isinstance(inp, np.ndarray)


def _is_list(inp):
    if isinstance(inp, np.ndarray):
        assert len(inp.shape) == 1
    else:
        assert isinstance(inp, list)


def square_box(img, box):
    _is_list(box)
    if isinstance(img, PIL.Image.Image):
        w, h = img.size
    elif isinstance(img, np.ndarray):
        h, w, c = img.shape
    else:
        raise TypeError('img is not valid. Expect `ndarray` or `PIL`')
    x, y, xx, yy = box
    _cx = (xx + x)/2
    _delta_y = (yy - y)/2
    x = int(np.max([0, _cx - _delta_y]))
    xx = int(np.min([w, _cx + _delta_y]))

    x = np.max([0, x])
    y = np.max([0, y])
    return [x, y, xx, yy]


def crop_img(img, box):
    mode = _get_type(img)
    if mode == 'np':
        return np_crop_img(img, box)
    elif mode == 'pil':
        return pil_crop_img(img, box)
    else:
        raise ValueError('Unknown mode!')


def np_crop_img(img_arr, box):
    x, y, xx, yy = box
    return img_arr[y:yy, x:xx, ...]


def pil_crop_img(img_pil, box):
    img_pil = img_pil.crop(box)
    return img_pil


def load_image_from_file(fpath, mode='rgb'):
    pil_img = Image.open(fpath)
    pil_img = pil_img.convert('RGB')
    return pil_img


def binarize_mask(img_arr, threshold=0.5):
    img_arr = img_arr.copy()
    img_arr[img_arr >= threshold] = 1
    img_arr[img_arr < threshold] = 0
    return img_arr.astype(np.uint8)


def pil_to_array(img_pil):
    return np.array(img_pil)


def array_to_pil(img_arr):
    return Image.fromarray(img_arr)


def gray_to_rgb(img_arr):
    return np.dstack([img_arr, img_arr, img_arr])


def rgb_to_gray(img_arr):
    return cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)


def expand_box(img, box, ratio=1.0):
    _is_list(box)
    mode = _get_type(img)
    if mode == 'np':
        w, h, c = img.shape
    elif mode == 'pil':
        w, h = img.size
    else:
        raise ValueError('Unknown mode!')
    x, y, xx, yy = box
    _w = xx - x
    delta_x = _w*ratio - _w
    x -= delta_x
    y -= delta_x
    xx += delta_x
    yy += delta_x
    x = int(np.max([0, x]))
    y = int(np.max([0, y]))
    xx = int(np.max([0, xx]))
    yy = int(np.max([0, yy]))
    return [x, y, xx, yy]


# PIL Function
def center_crop(img_pil, ratio=0.1):
    w, h = img_pil.size
    h_ = int(ratio*h)
    w_ = int(ratio*w)
    return img_pil.crop([w_, h_, h-h_, w-w_])


def random_shift_box(img_pil, bbox, ratio=0.2):
    w = bbox[2] - bbox[0]
    delta_x = np.random.randint(-int(ratio*w), int(ratio*w))
    delta_y = np.random.randint(-int(ratio*w), int(ratio*w))
    bbox[0] -= delta_x
    bbox[2] -= delta_x
    bbox[1] -= delta_y
    bbox[3] -= delta_y
    bbox = square_box(img_pil, bbox)
    return bbox


def random_rotate_with_box(img_pil, box, rot_range=[-10, 10], debug=False):
    minn = rot_range[0]
    maxx = rot_range[1]
    x, y, xx, yy = box
    center_x = (x + xx)//2
    center_y = (y + yy)//2
    img_pil = img_pil.rotate(np.random.randint(
        minn, maxx), center=(center_x, center_y))
    img_pil_draw = ImageDraw.Draw(img_pil)
    if debug:
        img_pil_draw.rectangle(([x, y, xx, yy]))
    return img_pil
