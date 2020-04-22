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
# python extract.py -f video_frame -p midia/main/1234/016-2017.mp4
# python extract.py -f video_frame_crop -p midia/main/1234/016-2017.mp4
# python extract.py -f stitch -p midia/main/1234/frame/016-2017
# python extract.py -f cryptometry -p midia/main/1234/stitch100.tif


import cv2 as cv
import numpy as np
import argparse
import subprocess
import pathlib
import sys
import math


def dir_structure(path, dir_list):
    for dire in dir_list:
        path_dir = path.parents[0] / dire
        if not path_dir.is_dir():
            path_dir.mkdir()
        sub_dir = path_dir / path.stem
        dir_exists(sub_dir)
        sub_dir.mkdir()
        print("New directory structure was created! Source: %s" % str(sub_dir))


def dir_exists(path):
    if path.is_dir():
        option = input(
            "Path %s already exists! Want to send to sandbox? (y/n) Caution:"
            " to press n will overwrite directory\n" % str(path))
        if option == "y":
            if "main" in str(path):
                hierarchy = path.parts
                main_index = hierarchy.index("main")
                path_index = len(hierarchy)-(2 + max(0, main_index-1))
                print("Directory was sent to sandbox! Code: %s" %
                      send_sandbox(path, (path.parents[path_index] / "sandbox" / hierarchy[main_index+1])))
            else:
                print("Directory 'main' not found! Exiting...")
                sys.exit()
        elif option == "n":
            subprocess.run("rm -rf "+str(path.resolve()), shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            print("Directory %s was deleted!" % str(path))
        else:
            print("Option unavailable! Exiting...")
            sys.exit()


def send_sandbox(path, dest_path):
    if not dest_path.is_dir():
        dest_path.mkdir()
    count = subprocess.run("find . -maxdepth 1 -type f | wc -l", cwd=dest_path,
                           shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    key_sand = '{:04d}'.format(int(count.stdout))
    subprocess.run("zip -r "+key_sand+".zip "+str(path),
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    subprocess.run("rm -rf "+str(path.resolve()),
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    mv = subprocess.run("mv -vn "+key_sand+".zip "+str(dest_path.resolve()),
                        shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if (mv.stdout == ''):
        print("Error to move: destination path already exists!")
    return key_sand


def video_frame(source, crop=False):
    # Convert a video to frame images
    path = pathlib.Path(source)
    dir_structure(path, ["frame"])
    sub_dir = path.parents[0] / "frame" / path.stem
    vidcap = cv.VideoCapture(source)
    success, image = vidcap.read()
    count = 0
    while success:
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = remove_text(gray_frame)
        if crop is True:
            image = image[75:500, 75:500]
        cv.imwrite(str(sub_dir)+"/frame%03d.png" % count, image)
        success, image = vidcap.read()
        count += 1
    print("Finished:", source)


def remove_text(image):
    # Remove white text from frame images
    cv.rectangle(image, (0, 0), (80, 30), (0, 0, 0), -1)
    cv.rectangle(image, (496, 504), (576, 584), (0, 0, 0), -1)
    return image


def cryptometry(source):
    path = pathlib.Path(source)
    dir_list = ["fig", "plot"]
    dir_structure(path, dir_list)
    print("Initialize cryptometry")
    image = cv.imread(source)
    cryptometry = []
    # cryptometry.append(axis_ratio(image.copy()))
    # cryptometry.append(perimeter(image.copy()))
    # cryptometry.append(mean_distance(image.copy()))
    cryptometry.append(wall_thickness(image.copy()))
    # print("\nParameters\t\t MEAN\t\t STD")
    # print("Axis ratio\t\t %.2f\t\t %.2f" %
    #       (cryptometry[0][0], cryptometry[0][1]))
    # print("Perimeter(px)\t\t %.2f\t %.2f" %
    #       (cryptometry[1][0], cryptometry[1][1]))
    # print("Sphericity(%%)\t\t %.2f\t\t %.2f" %
    #       (cryptometry[1][2], cryptometry[1][3]))
    # print("Mean distance(px)\t %.2f\t\t %.2f" %
    #       (cryptometry[2][0], cryptometry[2][1]))
    # print("Min  distance(px)\t %.2f\t\t %.2f" %
    #       (cryptometry[2][2], cryptometry[2][3]))
    print("\nFinished cryptometry")
    for sub_dir in dir_list:
        subprocess.run("mv -vn *"+sub_dir+".png "+str(path.parents[0] / sub_dir / path.stem),
                       shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def wall_thickness(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.dilate(thresh, kernel, iterations=10)
    erosion = cv.erode(dilate, kernel, iterations=10)
    contours, hierarchy = cv.findContours(
        erosion, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    num_crypts = 0
    center_list = []
    boundary_list = []
    saved_contours = []
    for cont in contours:
        area = cv.contourArea(cont)
        if area > 10000 and area < 310000:
            M = cv.moments(cont)
            coord = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center_list.append(coord)
            cimg = np.zeros_like(gray)
            cv.drawContours(cimg, cont, -1, (255, 255, 255), 1)
            pts = np.where(cimg == 255)
            boundary_list.append(pts)
            saved_contours.append(cont)
            num_crypts += 1
    count = 1
    wall_list = []
    MAX_DIST = 735
    for first_point in center_list:
        min_dist = MAX_DIST
        for second_point in center_list[count:]:
            dist = distance(first_point, second_point)
            if dist < MAX_DIST:
                slope = ((second_point[1] - first_point[1]) /
                         (second_point[0] - first_point[0]))
                cimg = np.zeros_like(gray)
                cv.drawContours(
                    cimg, saved_contours[count-1], -1, (255, 255, 255), 1)
                cv.drawContours(
                    cimg, saved_contours[count], -1, (255, 255, 255), 1)
                cv.line(cimg, first_point, second_point,
                        (255, 255, 255), thickness=1)
                for coord in range(0, len(boundary_list[count-1][0])):
                    point = (boundary_list[count-1][1][coord],
                             boundary_list[count-1][0][coord])
                    if collinear(slope, first_point, point, slope):
                        if between_points(first_point, second_point, point, slope):
                            # talvez possa dar break aqui
                            cv.circle(cimg,  point, 3, (255, 255, 255), -1)
                for coord in range(0, len(boundary_list[count][0])):
                    point = (boundary_list[count][1][coord],
                             boundary_list[count][0][coord])
                    if collinear(slope, first_point, point, slope):
                        if between_points(first_point, second_point, point, slope):
                            # talvez possa dar break aqui
                            cv.circle(cimg,  point, 3, (255, 255, 255), -1)
                cv.imwrite("wall.png", cimg)
                sys.exit()
                if dist < min_dist:
                    min_dist = dist


def between_points(first_point, second_point, mid_point, epsilon):
    return math.isclose((distance(first_point, mid_point)+distance(mid_point, second_point)), distance(first_point, second_point), abs_tol=epsilon)


def collinear(slope, first_point, collinear_point, epsilon):
    equation = ((slope*collinear_point[0]) -
                (slope*first_point[0]))+first_point[1]
    if math.isclose(equation, collinear_point[1], abs_tol=epsilon):
        return True
    return False


def distance(first_point, second_point):
    return np.sqrt(
        np.sum((np.subtract(first_point, second_point))**2))


def mean_distance(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.dilate(thresh, kernel, iterations=10)
    erosion = cv.erode(dilate, kernel, iterations=10)
    contours, hierarchy = cv.findContours(
        erosion, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    num_crypts = 0
    center_list = []
    for cont in contours:
        area = cv.contourArea(cont)
        if area > 10000 and area < 310000:
            M = cv.moments(cont)
            coord = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center_list.append(coord)
            cv.circle(image,  (coord), 7, (0, 0, 255), -1)
            num_crypts += 1
    count = 1
    mean_dist_list = []
    min_dist_list = []
    MAX_DIST = 735
    for first_point in center_list:
        min_dist = MAX_DIST
        for second_point in center_list[count:]:
            dist = distance(first_point, second_point)
            if dist < MAX_DIST:
                if dist < min_dist:
                    min_dist = dist
                cv.line(image, first_point, second_point,
                        (0, 0, 255), thickness=3)
                # print("Ponto %s\t M %s\t D %s" % (count, min_dist, dist))
                mean_dist_list.append(dist)
        if min_dist != MAX_DIST:
            min_dist_list.append(min_dist)
        count += 1
    print("Number of crypts assessed:", num_crypts)
    cv.imwrite("dist_fig.png", image)
    return np.mean(mean_dist_list), np.std(mean_dist_list), np.mean(min_dist_list), np.std(min_dist_list)


def perimeter(image):
    print("Initialize Perimeter")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
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
    cv.imwrite("perim_fig.png", image)
    return np.mean(perim_list), np.std(perim_list), np.mean(spher_list)*100, np.std(spher_list)*100


def axis_ratio(image):
    # Ratio between major and minor axis (Ma/ma ratio)
    # Give the mean and standard deviation of the ratio between the width and
    # the heigth of the box containing the crypt
    print("Initialize axis ratio")
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
    cv.imwrite("axisr_fig.png", image)
    print("Number of crypts assessed:", num_crypts)
    return np.mean(mama_list), np.std(mama_list)


# def stitch_stack(source):
#     # Stitch frame images to do a mosaic
#     list_images = sorted(os.listdir(source))
#     images = []
#     for image_name in list_images:
#         image = cv.imread(source+'/'+image_name)
#         images.append(image)
#         stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)
#         status, pano = stitcher.stitch(images)
#     if status != cv.Stitcher_OK:
#         print("Can't stitch images, error code = %d" % status)
#         sys.exit(-1)
#         cv.imwrite("teste.png", pano)
#         print("stitching completed successfully.")


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
    # elif (function == "stitch"):
    #     stitch_stack(source)
    elif (function == "cryptometry"):
        cryptometry(source)
    else:
        print("Undefined function")


if __name__ == "__main__":
    main()
