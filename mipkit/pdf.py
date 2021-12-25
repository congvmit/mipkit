import six
import pkg_resources
import sys
import re
import argparse
import os
from tqdm import tqdm
try:
    from PyPDF2 import PdfFileReader, PdfFileWriter
    from PyPDF2.pdf import PageObject
    # import PyPDF2 as fpdf
    from PyPDF2.generic import FloatObject
except ImportError:
    raise ImportError(
        'PyPDF2 requires to be installed in your machine. Please install it by `pip install PyPDF2`')

__VERSION__ = '1.0.0'


def version():
    return __VERSION__


def existance_verify(pdf_path) -> bool:
    return os.path.isfile(pdf_path)


def expand_margin(path_to_load, path_to_save=None, expected_margin=100, overwrite=False):
    folder_dir = os.path.dirname(path_to_load)
    filename = os.path.basename(path_to_load)
    base_filename, ext = os.path.splitext(filename)

    if not overwrite:
        new_filename = base_filename + '_expanded' + ext
    else:
        new_filename = filename
    path_to_save = os.path.join(folder_dir, new_filename)

    print('> Expand the margin for `{}`'.format(filename))

    with open(path_to_load, 'rb') as f:
        p = PdfFileReader(f)
        info = p.getDocumentInfo()
        number_of_pages = p.getNumPages()

        writer = PdfFileWriter()
        for i in tqdm(range(number_of_pages)):
            page = p.getPage(i)

            new_page = writer.addBlankPage(
                page.mediaBox.getWidth() + 2 * expected_margin,
                page.mediaBox.getHeight()
            )
            
            new_page.mergePage(page)

            # x, y, x1, y2
            bbox = new_page.mediaBox.getObject()
            bbox[0] = FloatObject(bbox[0] - expected_margin)
            bbox[2] = FloatObject(bbox[2] - expected_margin)
            
            # page.scaleTo(int(page.mediaBox.getWidth()),
            #              int(page.mediaBox.getHeight()))

            # new_page.mergeScaledTranslatedPage(page, scale=1,
            #                                    tx=expected_margin,
            #                                    ty=expected_margin)

            # writer.addPage(new_page)

        with open(path_to_save, 'wb') as f:
            writer.write(f)
    print('> Saved at `{}`'.format(path_to_save))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="display version",
    )
    parser.add_argument(
        "-p", "--path", required=True, help="url or file/folder id (with --id) to download from"
    )
    parser.add_argument("-o", "--output", help="output file name / path")
    parser.add_argument("--margin", help='margin size to expand', default=100)
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Show progress bar"
    )

    args = parser.parse_args()

    if args.version:
        print(version())
        exit()

    if args.path:
        exist = existance_verify(args.path)
        if not exist:
            print('File not Found')
            exit()
        else:
            expand_margin(path_to_load=args.path,
                          path_to_save=None,
                          expected_margin=100,
                          overwrite=False)

    # if args.folder:


if __name__ == "__main__":
    main()
