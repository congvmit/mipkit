import matplotlib.pyplot as plt
import cv2
import numpy as np


def draw_mask(img_arr, mask_arr):
    """Draw image with mask"""
    pass


def randint(val_min=0, val_max=255):
    return np.random.randint(val_min, val_max)


def draw_boxes(img_arr, bboxes, color=None, thickness=2):
    for box in bboxes:
        draw_box(img_arr, box, color=color, thickness=thickness)


def draw_box(img_arr, box, thickness=2, color=None):
    if color is None:
        color = (randint(), randint(), randint())
    x, y, xx, yy = box
    cv2.rectangle(img_arr, (x, y), (xx, yy), color=color, thickness=thickness)


def show_multi_images(list_img_arr, subplot_size=10, rows=1, plt_show=True, title=None):
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
    columns = len(list_img_arr)//rows
    fig = plt.figure(figsize=(subplot_size*columns, subplot_size))
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(list_img_arr[i - 1])
    if title is not None:
        plt.title(title)
    if plt_show:
        plt.show()
