# /home/congvm/Applications/itksnap-3.8.0-20190612-Linux-gcc64/lib/snap-3.8.0/ITK-SNAP

import subprocess
import os


class ITKSnap():
    def __init__(self, itk_snap_bin_path, show_command=True):
        """ITK Wrapper
            -g FILE              : Load the main image from FILE
            -s FILE              : Load the segmentation image from FILE
            -l FILE              : Load label descriptions from FILE
            -o FILE [FILE+]      : Load additional images from FILE
                                :   (multiple files may be provided)
            -w FILE              : Load workspace from FILE
                                :   (-w cannot be mixed with -g,-s,-l,-o options)
        Args:
            itk_snap_path (str): path to ITK-Snap
        """
        assert os.path.isfile(itk_snap_bin_path)
        self.itk_snap_bin_path = itk_snap_bin_path
        self.show_command = show_command

    def show(self, main_image_file_path, seg_image_file_path=None, add_main_image_file_path=None):
        assert os.path.isfile(main_image_file_path)

        command = [self.itk_snap_bin_path, '-g', main_image_file_path]

        if seg_image_file_path is not None:
            assert os.path.isfile(seg_image_file_path)
            command.append('-s')
            command.append(seg_image_file_path)

        if add_main_image_file_path is not None:
            assert os.path.isfile(add_main_image_file_path)
            command.append('-o')
            command.append(add_main_image_file_path)

        if self.show_command:
            print('Command: ', ' '.join(command))

        process = subprocess.run(command,
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
