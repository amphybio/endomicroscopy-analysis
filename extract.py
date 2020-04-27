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


def low_processing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # TODO 1) Improve parameters to get a more precise crypt sizes
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.dilate(thresh, kernel, iterations=10)
    erosion = cv.erode(dilate, kernel, iterations=10)
    processed_image = erosion
    return processed_image


def segmentation(image):
    processed_image = low_processing(image)
    contours_list, hierarchy = cv.findContours(
        processed_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    crypts_list = []
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > 10000 and area < 310000:
            crypts_list.append(countour)
    print("Number of crypts assessed:", len(crypts_list))
    return crypts_list


def cryptometry(source):
    path = pathlib.Path(source)
    dir_list = ["fig", "plot"]
    dir_structure(path, dir_list)
    print("Initialize cryptometry")
    image = cv.imread(source)
    crypts_list = segmentation(image.copy())
    cryptometry = []
    cryptometry.append(axis_ratio(image.copy(), crypts_list))
    cryptometry.extend(perimeter(image.copy(), crypts_list))
    cryptometry.extend(mean_distance(image.copy(), crypts_list))
    cryptometry.append(wall_thickness(image.copy(), crypts_list))
    print("\nParameters\t\t MEAN\t\t STD")
    print("Axis ratio\t\t %.2f\t\t %.2f" %
          (cryptometry[0][0], cryptometry[0][1]))
    print("Perimeter(px)\t\t %.2f\t %.2f" %
          (cryptometry[1][0], cryptometry[1][1]))
    print("Sphericity(%%)\t\t %.2f\t\t %.2f" %
          (cryptometry[2][0], cryptometry[2][1]))
    print("Mean distance(px)\t %.2f\t\t %.2f" %
          (cryptometry[3][0], cryptometry[3][1]))
    print("Min  distance(px)\t %.2f\t\t %.2f" %
          (cryptometry[4][0], cryptometry[4][1]))
    print("Wall Thickness(px)\t %.2f\t\t %.2f" %
          (cryptometry[5][0], cryptometry[5][1]))
    print("\nFinished cryptometry")
    for sub_dir in dir_list:
        subprocess.run("mv -vn *"+sub_dir+".png "+str(path.parents[0] / sub_dir / path.stem),
                       shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def wall_thickness(image, crypts_list):
    # Executando para os N primeiros vizinhos
    center_list = get_center(crypts_list)
    wall_list = []
    MAX_DIST = 735
    for first_index, first_point in enumerate(center_list):
        for second_index, second_point in enumerate(center_list[first_index+1:]):
            dist = distance(first_point, second_point)
            if dist < MAX_DIST:
                slope = ((second_point[1] - first_point[1]) /
                         (second_point[0] - first_point[0]))
                first_wall = []
                second_wall = []
                for point in crypts_list[first_index]:
                    point = point[0]
                    if collinear(slope, first_point, point):
                        if between_points(slope, first_point, second_point, point):
                            first_wall.append(point)
                nested_index = first_index + 1 + second_index
                for point in crypts_list[nested_index]:
                    point = point[0]
                    if collinear(slope, first_point, point):
                        if between_points(slope, first_point, second_point, point):
                            second_wall.append(point)
                min_wall = MAX_DIST
                minA = [1]
                minB = [1]
                for pointA in first_wall:
                    for pointB in second_wall:
                        dist = distance(pointA, pointB)
                        if dist < min_wall:
                            min_wall = dist
                            minA[0] = pointA
                            minB[0] = pointB
                    wall_list.append(min_wall)
                    min_wall = MAX_DIST
                cv.circle(image,  tuple(minA[0]), 7, (0, 0, 255), -1)
                cv.circle(image,  tuple(minB[0]), 7, (0, 0, 255), -1)
                cv.line(image, tuple(minA[0]), tuple(minB[0]), (0, 0, 255), 3)
    cv.imwrite("wall_fig.png", image)
    return np.mean(wall_list), np.std(wall_list)


def between_points(slope, first_point, second_point, mid_point):
    return math.isclose((distance(first_point, mid_point)+distance(mid_point, second_point)), distance(first_point, second_point), abs_tol=abs(slope))


def collinear(slope, first_point, collinear_point):
    equation = ((slope*collinear_point[0]) -
                (slope*first_point[0]))+first_point[1]
    return math.isclose(equation, collinear_point[1], abs_tol=(abs(slope))+1)


def mean_distance(image, crypts_list):
    center_list = get_center(crypts_list)
    for center in center_list:
        cv.circle(image,  (center), 7, (0, 0, 255), -1)
    mean_dist_list = []
    min_dist_list = []
    MAX_DIST = 735
    for index, first_point in enumerate(center_list):
        min_dist = MAX_DIST
        for second_point in center_list[index+1:]:
            dist = distance(first_point, second_point)
            if dist < MAX_DIST:
                cv.line(image, first_point, second_point,
                        (0, 0, 255), thickness=3)
                mean_dist_list.append(dist)
                if dist < min_dist:
                    min_dist = dist
        if min_dist != MAX_DIST:
            min_dist_list.append(min_dist)
    cv.imwrite("dist_fig.png", image)
    return [np.mean(mean_dist_list), np.std(mean_dist_list)], [np.mean(min_dist_list), np.std(min_dist_list)]


def distance(first_point, second_point):
    return np.sqrt(
        np.sum((np.subtract(first_point, second_point))**2))


def get_center(crypts_list):
    center_list = []
    for crypt in crypts_list:
        M = cv.moments(crypt)
        coord = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        center_list.append(coord)
    return center_list


def perimeter(image, crypts_list):
    perim_list = []
    spher_list = []
    for crypt in crypts_list:
        perimeter = cv.arcLength(crypt, True)
        perim_list.append(perimeter)
        area = cv.contourArea(crypt)
        spher_list.append((4 * np.pi * area) / (perimeter ** 2))
        # epsilon = 0.007 * cv.arcLength(cont, True)
        # approx = cv.approxPolyDP(cont, epsilon, True)
        # cv.drawContours(image, [approx], -1, (0, 255, 0), 3)
        cv.drawContours(image, crypt, -1, (0, 0, 255), 3)
    cv.imwrite("perim_fig.png", image)
    return [np.mean(perim_list), np.std(perim_list)], [np.mean(spher_list)*100, np.std(spher_list)*100]


def axis_ratio(image, crypts_list):
    # Ratio between major and minor axis (Ma/ma ratio)
    # Give the mean and standard deviation of the ratio between the width and
    # the heigth of the box containing the crypt
    mama_list = []
    for crypt in crypts_list:
        x, y, width, heigth = cv.boundingRect(crypt)
        if (width > heigth):
            mama_list.append(width/heigth)
        else:
            mama_list.append(heigth/width)
        cv.rectangle(image, (x, y), (x + width, y + heigth), (0, 0, 255), 3)
    cv.imwrite("axisr_fig.png", image)
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
