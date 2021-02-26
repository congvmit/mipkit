from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL


def read_image(file_path: str, to_rgb=True) -> np.ndarray:
    img_arr = cv2.imread(file_path)
    if to_rgb:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    return img_arr


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
