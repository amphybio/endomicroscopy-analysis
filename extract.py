#   Copyright (C) 2020 AMPhyBio
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with program.  If not, see <http://www.gnu.org/licenses/>.

# =============================================================================
#       ARQUIVO:  extract.py
#
#     DESCRIÇÃO: Script to extract features of endomicroscopy images
#
#        OPÇÕES:  ---
#    REQUISITOS:  OpenCV, Python, Numpy
#          BUGS:  ---
#         NOTAS:  ---
#         AUTOR:  Alan U. Sabino <alan.sabino@usp.br>
#        VERSÃO:  0.1
#       CRIAÇÃO:  14/02/2020
#       REVISÃO:  ---
# =============================================================================

# USAGE
# python extract.py -f video_frame -p midia/016-2017.mp4
# python extract.py -f video_frame_crop -p midia/016-2017.mp4
# python extract.py -f stitch -p midia/016-2017
# python extract.py -f stitch -p midia/tres

import cv2 as cv
import numpy as np
import argparse
import os
import sys
import shutil


def video_frame(source, crop=False):
    vidcap = cv.VideoCapture(source)
    success, image = vidcap.read()
    count = 0
    path = os.path.splitext(source)[0]
    is_dir(path)
    os.mkdir(path)
    while success:
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = remove_text(gray_frame)
        if (crop is True):
            image = image[75:500, 75:500]
        cv.imwrite(path+"/frame%03d.png" % count, image)
        success, image = vidcap.read()
        count += 1
    file = os.path.basename(source)
    print('Finished ', file)


def is_dir(source):
    isdir = os.path.isdir(source)
    if (isdir):
        option = input("Path "+source+" already exists! Want to Overwrite or save Backup? (o/b)\n")
        if (option == "b"):
            is_dir(source+".BKP")
            os.rename(source, source+'.BKP')
            print("Backup complete! Backup path: ", source+'.BKP')
        else:
            shutil.rmtree(source)
            print("Directory overwrited!")


def remove_text(image):
    cv.rectangle(image, (0, 0), (80, 30), (0, 0, 0), -1)
    cv.rectangle(image, (496, 504), (576, 584), (0, 0, 0), -1)
    return image


def stitch_stack(source):
    list_images = sorted(os.listdir(source))
    images = []
    for image_name in list_images:
        image = cv.imread(source+'/'+image_name)
        images.append(image)
    stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    cv.imwrite("teste.png", pano)
    print("stitching completed successfully.")


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--function", type=str, required=True, help="Set a function to call (video_frame; video_frame_crop)")
ap.add_argument("-p", "--path", type=str, required=False, help="Input file or directory of images path")
args = vars(ap.parse_args())

function = args["function"]
source = args["path"]

if (function == "video_frame"):
    video_frame(source)
elif (function == "video_frame_crop"):
    video_frame(source, True)
elif (function == "stitch"):
    stitch_stack(source)
else:
    print("Undefined function")
