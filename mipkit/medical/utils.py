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

import datetime
from pydicom.dataset import FileDataset, FileMetaDataset
import itk
import SimpleITK as sitk


def load_nii_file(path_to_load):
    sitk_image = sitk.ReadImage(path_to_load)
    return sitk.GetArrayFromImage(sitk_image)


def load_DICOM_series(folder_dir):
    # https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage/Documentation.html
    PixelType = itk.ctype("signed short")
    Dimension = 3

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

    seriesIdentifier = seriesUID[0]  # Get the first one
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)

    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()
    itk_image = reader.GetOutput()
    output_arr = itk.GetArrayFromImage(itk_image)
    return output_arr


def save_NII_file(image_3d, path_to_save):
    file_meta = FileMetaDataset()
    ds = FileDataset(path_to_save, {},
                     file_meta=file_meta)  # , preamble=b"\0" * 128)

    # transform array to 3D image
    ct_pet_image = sitk.GetImageFromArray(image_3d)

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%3f')  # long format with micro seconds
    ds.ContentTime = timeStr
    # ds.save_as(path_to_save, write_like_original=True)
    sitk.WriteImage(image_3d, path_to_save)  # , imageIO='NiftiImageIO')
