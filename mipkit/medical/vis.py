import matplotlib.pyplot as plt
import numpy as np
from .. import immulshow


def im3dshow(image_3d, slide_x_pos=0, slide_y_pos=0, slide_z_pos=0, ratio_size=10, **kwargs):
    slide_x_view = image_3d[slide_x_pos, :, :]
    slide_y_view = image_3d[:, slide_y_pos, :][::-1]
    slide_z_view = image_3d[:, :, slide_z_pos][::-1]
    return immulshow([slide_x_view, slide_y_view, slide_z_view],
                     ratio_size=ratio_size,
                     **kwargs)
