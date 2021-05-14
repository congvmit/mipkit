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

import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2
import numpy as np

from .images import read_image
from .utils import deprecated

TEXT_COLOR = (255, 255, 255)


def show_image(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img)


def get_random_rgb():
    return (np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256))


def visualize_bbox(img, bbox, class_name=None, color=None, thickness=2):
    """Visualizes a single bounding box on the image"""
    color = color or get_random_rgb()
    if isinstance(bbox, list): 
        bbox = np.array(bbox)
    x_min, y_min, x_max, y_max = bbox.astype(int)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)

    if class_name is not None:
        ((text_width, text_height), _) = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

        cv2.rectangle(img, (x_min, y_min - int(2.0 * text_height)),
                      (x_min + int(text_width*1.5), y_min), color, -1)

        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
    return img


def convert_coco_to_default(bbox):
    '''x, y, w, h --> x, y, xx, yy'''
    bbox_coco = bbox.copy()
    bbox_coco[:, 2] = bbox_coco[:, 2] + bbox_coco[:, 0]
    bbox_coco[:, 3] = bbox_coco[:, 3] + bbox_coco[:, 1]
    return bbox_coco


def convert_default_to_coco(bbox):
    '''x, y, xx, yy --> x, y, w, h'''
    bbox_coco = bbox.copy()
    bbox_coco[:, 2] = bbox_coco[:, 2] - bbox_coco[:, 0]
    bbox_coco[:, 3] = bbox_coco[:, 3] - bbox_coco[:, 1]
    return bbox_coco


def visualize(image, bboxes,
              scores=None,
              category_ids=None,
              category_id_to_name=None,
              figsize=(12, 12),
              thickness=2,
              color=None,
              bbox_format='default',
              title=None,
              fontsize=15):
    img = image.copy()

    scores = [''] * len(bboxes) if scores is None else scores
    category_ids = [''] * len(bboxes) if category_ids is None else category_ids

    for bbox, sc, category_id in zip(bboxes, scores, category_ids):
        if category_id_to_name is not None:
            class_name = category_id_to_name[category_id]
            class_name = class_name + ' - ' + str(np.round(sc, 3))
        else:
            class_name = None
        img = visualize_bbox(img, bbox, class_name,
                             color=color,
                             thickness=thickness)

    plt.figure(figsize=figsize)
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.imshow(img)


def draw_mask(img_arr, mask_arr):
    """Draw image with mask"""
    pass


def randint(val_min=0, val_max=255):
    return np.random.randint(val_min, val_max)


@deprecated(message='draw_boxes() function is deprecated. Please use visualize() instead.')
def draw_boxes(img_arr, bboxes, color=None, thickness=2, mode='default'):
    assert mode in ['default', 'coco']
    for box in bboxes:
        draw_box(img_arr, box, color=color, thickness=thickness, mode=mode)


@deprecated(message='draw_box() function is deprecated. Please use visualize() instead.')
def draw_box(img_arr, box, thickness=2, color=None, mode='default'):
    h_img, w_img, c_img = img_arr.shape
    if color is None:
        color = (randint(), randint(), randint())
    if mode == 'default':
        x, y, xx, yy = box
        x = int(max(0, x))
        xx = int(min(w_img, xx))
        y = int(max(0, y))
        yy = int(min(h_img, yy))

    elif mode == 'coco':
        x, y, w, h = box
        x = int(max(0, x))
        w = int(min(w_img, x+w_img))
        y = int(max(0, y))
        h = int(min(h_img, y+h_img))
        x, y, xx, yy = x, y, x+w, y+h

    cv2.rectangle(img_arr, (x, y), (xx, yy), color=color, thickness=thickness)


def show_multi_images(list_img_arr,
                      list_titles=None,
                      ratio_size=10,
                      rows=1,
                      cmap=None,
                      plt_show=True,
                      title=None,
                      show_colorbar=False,
                      colorbar_fontsize=20,
                      fontsize=30,
                      wspace=0,
                      hspace=0, *args, **kwargs):
    """Show multiple images in a plot.

    Parameters
    ----------
    list_img_arr : list
        a list of numpy arrays
    subplot_size: int
        subplot image size to show
    rows: int
        A number of rows to show images
    plt_show: bool
        Finalize preparing plot and display
    title: str
        A title of the figure
    """

    assert ratio_size >= 2, ValueError("ratio_size must be greater than 1")
    columns = len(list_img_arr)//rows

    fig = plt.figure(figsize=(int(ratio_size*columns),
                              int((ratio_size/2)*rows)))

    gs = gridspec.GridSpec(rows, columns,
                           wspace=wspace,
                           hspace=wspace)

    for i in range(1, columns*rows + 1):
        a = fig.add_subplot(rows, columns, i)
        plt.imshow(list_img_arr[i - 1], cmap=cmap)
        a.set_aspect('equal')
        a.set_xticklabels([])
        a.set_yticklabels([])

        if list_titles is not None:
            a.set_title(list_titles[i - 1])

    if show_colorbar:
        cbar = plt.colorbar()
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(colorbar_fontsize)

    if title:
        fig.suptitle(title, fontsize=fontsize)

    if plt_show:
        plt.show()


def show_image_with_path(path, img_dir=None):
    if img_dir:
        path = os.path.join(img_dir, path)
    img_arr = read_image(path)
    return img_arr


def show_image_with_paths(list_paths,
                          rows=1,
                          img_dir=None, **kwargs):
    list_img_arr = []
    for path in list_paths:
        img_arr = show_image_with_path(path, img_dir=img_dir)
        list_img_arr.append(img_arr)
    show_multi_images(list_img_arr=list_img_arr, rows=rows, **kwargs)
