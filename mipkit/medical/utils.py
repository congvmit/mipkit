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

import nibabel as nib
import datetime
from pydicom.dataset import FileDataset, FileMetaDataset
import itk
import SimpleITK as sitk
import numpy as np
import cv2
from ..utils import deprecated


def flip_diagonally(x):
    """Diagonally flip all slides in a medical image

    Args:
        x (sitk.Image): MedicalImage format

    Returns:
        x: sitk.Image - MedicalImage format
    """
    #! TODO: Optimize process time
    x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    x = cv2.flip(x, 0)
    return x


@deprecated('Please use `mipkit.medical.load_medical_image_file` instead')
def load_nii_file(path_to_load: str):
    return load_medical_image_from_file(path_to_load)


def load_medical_image_from_file(path_to_load: str, return_array_only=True):
    """Load medical file

    Args:
        path_to_load (str): Path to medical image file

    Returns:
        ndarray: Numpy array
    """
    # sitk_image = sitk.ReadImage(path_to_load)
    # return sitk.GetArrayFromImage(sitk_image)
    nib_obj = nib.load(path_to_load)
    if return_array_only:
        return nib_obj.get_data()
    else:
        return nib_obj


def cvt_array_to_ITKImage(np_image: np.ndarray) -> sitk.Image:
    """Convert numpy array to sitk.Image

    Args:
        np_image (np.ndarray): Input Numpy Array

    Returns:
        sitk.Image: MedicalImage file
    """
    return sitk.GetImageFromArray(np_image)


def load_DICOM_series(folder_dir: str, series_index: int = 0, Dimension: int = 3):
    # https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage/Documentation.html
    PixelType = itk.ctype("signed short")
    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(folder_dir)

    # Beware, seriesUID is a tuple
    # Example:
    # ('1.2.840.113619.2.379.114374080023902.100461.1600744141171.5.33.75000051251220200825',)
    seriesUID = namesGenerator.GetSeriesUIDs()

    seriesIdentifier = seriesUID[series_index]  # Get the first one
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)

    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()
    itk_image = reader.GetOutput()
    output_arr = itk.GetArrayFromImage(itk_image)
    return output_arr


@ deprecated("Please use `mipkit.medical.save_medical_image_file` instead!")
def save_3d_file(image_3d, path_to_save):
    return save_medical_image_file(image_3d, path_to_save)


def save_medical_image_file(image_3d: sitk.Image, path_to_save: str):
    file_meta = FileMetaDataset()
    ds = FileDataset(path_to_save, {},
                     file_meta=file_meta)  # , preamble=b"\0" * 128)

    # transform array to 3D image
    image_3d = sitk.GetImageFromArray(image_3d)

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%3f')  # long format with micro seconds
    ds.ContentTime = timeStr
    # ds.save_as(path_to_save, write_like_original=True)
    sitk.WriteImage(image_3d, path_to_save)  # , imageIO='NiftiImageIO')
