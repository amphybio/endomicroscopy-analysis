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
    print("\nParameters\t MEAN\t\t STD")
    print("Ma/ma ratio\t %.5f\t %.5f" % (cryptometry[0][0], cryptometry[0][1]))
    print("\nFinished cryptometry")


def perimeter(image):
    # return hough_circles(image)
    return find_contours(image)


def find_contours(image):
    print("Initialize Perimeter")
    canvas = np.zeros(image.shape, np.uint8)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Smoothing
    # kernel = np.ones((5, 5), np.float32)/25
    # gray = cv.filter2D(gray, -1, kernel)

    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # im2, contours, hierarchy = cv.findContours(
    #     thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    contours, a = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    # print(cnt)
    # cv.imwrite("perm.png", cnt)

    num_crypts = 0
    if cnt is not None:
        max_area = cv.contourArea(cnt)
        print(max_area)
        for cont in contours:
            #max_area = cv.contourArea(cont)
            if cv.contourArea(cont) > max_area:
                num_crypts += 1
                cnt = cont
                max_area = cv.contourArea(cont)

    perimeter = cv.arcLength(cnt, True)
    epsilon = 0.01*cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    hull = cv.convexHull(cnt)

    cv.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
    cv.drawContours(canvas, [approx], -1, (0, 0, 255), 3)
    # cv.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.

    cv.imshow("Contour", canvas)
    k = cv.waitKey(0)

    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    return 1


def hough_circles(image):
    print("Initialize Perimeter")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,
                              1.5, 185, param1=20, param2=150, minRadius=95, maxRadius=185)

    num_crypts = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for(x, y, r) in circles:
            num_crypts += 1
            cv.circle(image, (x, y), r, (0, 0, 255), 3)
            cv.rectangle(image, (x-5, y-5), (x+5, y+5), (0, 255, 0), -1)

    cv.imwrite("perm.png", image)
    print("Number of crypts assessed:", num_crypts)
    return 1  # np.mean(mama_list), np.std(mama_list)


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
    else:
        print("Undefined function")


if __name__ == "__main__":
    main()
