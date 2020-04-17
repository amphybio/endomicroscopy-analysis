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
#          FILE:  extract.py
#
#   DESCRIPTION: Script to extract features of endomicroscopy images
#
#       OPTIONS:   -f FUNCTION, --function FUNCTION
#                     Set a function to call (video_frame; video_frame_crop,
#                       cryptometry)
#                  -p PATH, --path PATH
#                     Input file or directory of images path
#  REQUIREMENTS:  OpenCV, Python, Numpy
#          BUGS:  ---
#         NOTES:  ---
#         AUTOR:  Alan U. Sabino <alan.sabino@usp.br>
#       VERSION:  0.2
#       CREATED:  14/02/2020
#      REVISION:  ---
# =============================================================================

# USAGE
# python extract.py -f dir_structure -p midia/main/1234
# python extract.py -f video_frame -p midia/016-2017.mp4
# python extract.py -f video_frame_crop -p midia/016-2017.mp4
# python extract.py -f stitch -p midia/016-2017
# python extract.py -f stitch -p midia/tres
# python extract.py -f cryptometry -p midia/stitch100.tif
# python extract.py -f cryptometry -p midia/stitch300.tif


import cv2 as cv
import numpy as np
import argparse
import os
import sys
import shutil
import subprocess
import pathlib


def dir_structure(source):  # midia/main/1234
    path = pathlib.Path(source)
    if (path.is_dir()):
        option = input(
            "Path "+source+" already exists! Want to send to sandbox? (y/n)\n")
        if (option == "y"):
            print("Directory sent to sandbox! Code:", send_sandbox(source))
        else:
            print("Nothing to do here!")
    else:
        path.mkdir()
        for struct in {"frames", "figs", "plots", "videos"}:
            pathlib.Path(source+"/"+struct).mkdir()
        print("Directory structure created! Source: "+source)


def send_sandbox(source):
    path = pathlib.Path(source)
    id = path.name
    dest_path = path.parent.parent / "sandbox" / id

    if not dest_path.is_dir():
        dest_path.mkdir()

    count = subprocess.run("find . -maxdepth 1 -type f | wc -l", cwd=dest_path,
                           shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    id_sand = '{:04d}'.format(int(count.stdout))

    subprocess.run("zip -r "+id_sand+".zip frames/ figs/ plots/", cwd=path,
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    mv = subprocess.run("mv -vn "+id_sand+".zip "+str(dest_path.absolute()), cwd=path,
                        shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    if (mv.stdout == '') or (mv.stderr is not None):
        print("Error: file already exists or exiting with error:", mv.stderr)
        quit()

    subprocess.run("rm -rf frames/* figs/* plots/*", cwd=path,
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    return id_sand


def video_frame(source, crop=False):
    # Convert a video to frame images
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
    # Verify if a path is a directory and if it already exists
    isdir = os.path.isdir(source)
    if (isdir):
        option = input(
            "Path "+source+" already exists! Want to Overwrite or save Backup? (o/b)\n")
        if (option == "o"):
            shutil.rmtree(source)
            print("Directory overwrited!")
        else:
            is_dir(source+".BKP")
            os.rename(source, source+'.BKP')
            print("Backup complete! Backup path: ", source+'.BKP')


def remove_text(image):
    # Remove white text from frame images
    cv.rectangle(image, (0, 0), (80, 30), (0, 0, 0), -1)
    cv.rectangle(image, (496, 504), (576, 584), (0, 0, 0), -1)
    return image


def cryptometry(source):
    # TODO 1) Create an array with the names of functions and call them in a FOR
    # loop, print results and create/update a file with measurements; TODO 2)
    # Get a list of images in a source and make the measurements with all of
    # them
    print("Initialize cryptometry")
    image = cv.imread(source)
    cryptometry = []
    cryptometry.append(mama_ratio(image.copy()))
    cryptometry.append(perimeter(image.copy()))
    print("\nParameters\t MEAN\t STD")
    print("Ma/ma ratio\t %.2f\t %.2f" %
          (cryptometry[0][0], cryptometry[0][1]))
    print("Perimeter(px)\t %.2f\t %.2f" %
          (cryptometry[1][0], cryptometry[1][1]))
    print("Sphericity(%%)\t %.2f\t %.2f" %
          (cryptometry[1][2], cryptometry[1][3]))
    print("\nFinished cryptometry")


def perimeter(image):
    print("Initialize Perimeter")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # TODO 1) Improve parameters to get a more precise crypt sizes
    dilate = cv.dilate(thresh, kernel, iterations=10)
    erosion = cv.erode(dilate, kernel, iterations=10)

    contours, hierarchy = cv.findContours(
        erosion, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    cnt = contours[0]
    num_crypts = 0
    perim_list = []
    spher_list = []
    if cnt is not None:
        for cont in contours:
            area = cv.contourArea(cont)
            if area > 10000 and area < 310000:
                perimeter = cv.arcLength(cont, True)
                perim_list.append(perimeter)
                spher_list.append((4 * np.pi * area) / (perimeter ** 2))
                num_crypts += 1
                # epsilon = 0.007 * cv.arcLength(cont, True)
                # approx = cv.approxPolyDP(cont, epsilon, True)
                # cv.drawContours(image, [approx], -1, (0, 255, 0), 3)
                cv.drawContours(image, cont, -1, (0, 0, 255), 3)

    print("Number of crypts assessed:", num_crypts)
    cv.imwrite("perm.png", image)
    return np.mean(perim_list), np.std(perim_list), np.mean(spher_list)*100, np.std(spher_list)*100


def mama_ratio(image):
    # Major axis/minor axis ratio (Ma/ma ratio)
    # Give the mean and standard deviation of the ratio between the width and
    # the heigth of the box containing the crypt
    print("Initialize Major axis/Minor axis ratio")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # TODO 1) Improve parameters to get a more precise crypt sizes
    dilate = cv.dilate(thresh, kernel, iterations=10)
    erosion = cv.erode(dilate, kernel, iterations=10)

    aux = erosion

    cnts = cv.findContours(aux, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    mama_list = []
    num_crypts = 0
    for c in cnts:
        area = cv.contourArea(c)
        if area > 10000 and area < 310000:
            num_crypts += 1
            x, y, width, heigth = cv.boundingRect(c)
            if (width > heigth):
                mama_list.append(width/heigth)
            else:
                mama_list.append(heigth/width)
            cv.rectangle(image, (x, y), (x + width, y + heigth),
                         (0, 0, 255), 3)
    cv.imwrite("mama.png", image)
    print("Number of crypts assessed:", num_crypts)
    return np.mean(mama_list), np.std(mama_list)


def stitch_stack(source):
    # Stitch frame images to do a mosaic
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--function", type=str, required=True,
                    help="Set a function to call (video_frame; video_frame_crop, cryptometry)")
    ap.add_argument("-p", "--path", type=str, required=False,
                    help="Input file or directory of images path")
    args = vars(ap.parse_args())

    function = args["function"]
    source = args["path"]

    if (function == "video_frame"):
        video_frame(source)
    elif (function == "video_frame_crop"):
        video_frame(source, True)
    elif (function == "stitch"):
        stitch_stack(source)
    elif (function == "cryptometry"):
        cryptometry(source)
    elif (function == "dir_structure"):
        dir_structure(source)
    else:
        print("Undefined function")


if __name__ == "__main__":
    main()
