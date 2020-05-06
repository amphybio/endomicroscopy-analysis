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
#  REQUIREMENTS:  OpenCV, Python, Numpy, Seaborn
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
from timeit import default_timer as timer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Define style plots
sns.set(context='notebook', style='ticks', font='Arial', font_scale=2, rc={
        'axes.grid': True, 'grid.linestyle': 'dashed', 'lines.linewidth': 2, 'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.figsize': (7, 3.09017)})  # (1.4, 0.618034)
sns.set_palette(palette='bright')


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
            " Path %s already exists! Want to send to sandbox? (y/n)"
            " *Caution!* To press n will overwrite directory\n" % str(path))
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
    MIN_AREA = 10000
    MAX_AREA = 310000
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
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
    draw_countours(image, crypts_list)
    crypts_measures = []
    time = []
    start = timer()
    crypts_measures.append(axis_ratio(image.copy(), crypts_list))
    end = timer()
    time.append(end-start)
    start = timer()
    crypts_measures.extend(perimeter(image.copy(), crypts_list))
    end = timer()
    time.append(end-start)
    start = timer()
    crypts_measures.extend(mean_distance(image.copy(), crypts_list))
    end = timer()
    time.append(end-start)
    start = timer()
    crypts_measures.append(wall_thickness(image.copy(), crypts_list, 'H'))
    # crypts_measures.append(wall_thickness(image.copy(), crypts_list))
    end = timer()
    time.append(end-start)
    start = timer()
    crypts_measures.append(maximal_feret(image.copy(), crypts_list, 'H'))
    #crypts_measures.append(maximal_feret(image.copy(), crypts_list))
    end = timer()
    time.append(end-start)
    print("\nMeasures\t\t MEAN\t\t STD\t\t TIME(s)")
    print("Axis ratio\t\t %.2f\t\t %.2f\t\t %.2f" %
          (crypts_measures[0][0], crypts_measures[0][1], time[0]))
    print("Perimeter(\u03BCm)\t\t %.2f\t\t %.2f\t\t %.2f" %
          (crypts_measures[1][0], crypts_measures[1][1], time[1]))
    print("Sphericity(%%)\t\t %.2f\t\t %.2f\t\t -" %
          (crypts_measures[2][0], crypts_measures[2][1]))
    print("Mean distance(\u03BCm)\t %.2f\t\t %.2f\t\t %.2f" %
          (crypts_measures[3][0], crypts_measures[3][1], time[2]))
    print("Min  distance(\u03BCm)\t %.2f\t\t %.2f\t\t -" %
          (crypts_measures[4][0], crypts_measures[4][1]))
    print("Wall Thickness(\u03BCm)\t %.2f\t\t %.2f\t\t %.2f" %
          (crypts_measures[5][0], crypts_measures[5][1], time[3]))
    print("Max Feret(\u03BCm)\t\t %.2f\t\t %.2f\t\t %.2f" %
          (crypts_measures[6][0], crypts_measures[6][1], time[4]))
    print("\nFinished cryptometry")
    for sub_dir in dir_list:
        subprocess.run("mv -vn *"+sub_dir+".jpg "+str(path.parents[0] / sub_dir / path.stem),
                       shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def neighbors(crypts_list):
    MAX_DIST = 735
    neighbors_list = [[] for crypt in range(len(crypts_list))]
    center_list = get_center(crypts_list)
    for crypt_index, first_center in enumerate(center_list):
        for neighbor_index, second_center in enumerate(center_list):
            dist = distance(first_center, second_center)
            if dist < MAX_DIST and dist != 0:
                neighbors_list[crypt_index].append((
                    neighbor_index, dist))
    return neighbors_list


def maximal_feret(image, crypts_list, algorithm='B'):
    feret_diameters = []
    if algorithm == 'B':
        # BRUTE-FORCE
        for crypt in crypts_list:
            max_dist = 0
            max_pointA = [0]
            max_pointB = [0]
            for index, pointA in enumerate(crypt):
                pointA = pointA[0]
                for pointB in crypt[index+1:]:
                    pointB = pointB[0]
                    dist = distance(pointA, pointB)
                    if dist > max_dist:
                        max_dist = dist
                        max_pointA[0] = pointA
                        max_pointB[0] = pointB
            cv.circle(image, tuple(max_pointA[0]), 7, (0, 0, 255), -1)
            cv.circle(image, tuple(max_pointB[0]), 7, (0, 0, 255), -1)
            cv.line(image, tuple(max_pointA[0]), tuple(
                max_pointB[0]), (0, 0, 255), thickness=3)
            feret_diameters.append(max_dist)
    else:
        # HEURISTIC
        for crypt in crypts_list:
            left = tuple(crypt[crypt[:, :, 0].argmin()][0])
            right = tuple(crypt[crypt[:, :, 0].argmax()][0])
            top = tuple(crypt[crypt[:, :, 1].argmin()][0])
            bottom = tuple(crypt[crypt[:, :, 1].argmax()][0])
            y_distance = distance(top, bottom)
            x_distance = distance(left, right)
            if x_distance > y_distance:
                cv.circle(image, left, 7, (0, 255, 255), -1)
                cv.circle(image, right, 7, (0, 255, 255), -1)
                cv.line(image, left, right, (0, 255, 255), thickness=3)
                feret_diameters.append(x_distance)
            else:
                cv.circle(image, top, 7, (0, 255, 255), -1)
                cv.circle(image, bottom, 7, (0, 255, 255), -1)
                cv.line(image, top, bottom, (0, 255, 255), thickness=3)
                feret_diameters.append(y_distance)
    cv.imwrite("feret_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    feret_diameters = pixel_micrometer(feret_diameters)
    return np.mean(feret_diameters), np.std(feret_diameters)


def wall_thickness(image, crypts_list, algorithm='B'):
    MAX_DIST = 735
    wall_list = [0] * len(crypts_list)
    neighbors_list = neighbors(crypts_list)
    for crypt_index, crypt in enumerate(crypts_list):
        min_wall = MAX_DIST
        wall_crypt_point = [0]
        wall_neighbor_point = [0]
        for neighbor in neighbors_list[crypt_index]:
            if algorithm == 'B':
                # BRUTE-FORCE
                for crypt_point in crypt:
                    for neighbor_point in crypts_list[neighbor[0]]:
                        dist_wall = distance(crypt_point[0], neighbor_point[0])
                        if dist_wall < min_wall:
                            min_wall = dist_wall
                            wall_crypt_point[0] = crypt_point[0]
                            wall_neighbor_point[0] = neighbor_point[0]
                            wall_list[crypt_index] = min_wall
            else:
                # HEURISTIC
                center_list = get_center(crypts_list)
                crypt_center = center_list[crypt_index]
                neighbor_center = center_list[neighbor[0]]
                slope = ((neighbor_center[1] - crypt_center[1]) /
                         (neighbor_center[0] - crypt_center[0]))
                crypt_wall = []
                for crypt_point in crypt:
                    if collinear(slope, crypt_center, crypt_point[0]):
                        if between_points(slope, crypt_center, neighbor_center, crypt_point[0]):
                            crypt_wall.append(crypt_point[0])
                neighbor_wall = []
                for neighbor_point in crypts_list[neighbor[0]]:
                    if collinear(slope, crypt_center, neighbor_point[0]):
                        if between_points(slope, crypt_center, neighbor_center, neighbor_point[0]):
                            neighbor_wall.append(neighbor_point[0])
                for pointA in crypt_wall:
                    for pointB in neighbor_wall:
                        dist_wall = distance(pointA, pointB)
                        if dist_wall < min_wall:
                            min_wall = dist_wall
                            wall_crypt_point[0] = pointA
                            wall_neighbor_point[0] = pointB
                            wall_list[crypt_index] = min_wall
        cv.circle(image,  tuple(wall_crypt_point[0]), 7, (0, 0, 255), -1)
        cv.circle(image,  tuple(wall_neighbor_point[0]), 7, (0, 0, 255), -1)
        cv.line(image, tuple(wall_crypt_point[0]), tuple(
            wall_neighbor_point[0]), (0, 0, 255), 3)
    cv.imwrite("wall_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    wall_list = pixel_micrometer(wall_list)
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
    cv.imwrite("dist_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    mean_dist_list = pixel_micrometer(mean_dist_list)
    min_dist_list = pixel_micrometer(min_dist_list)
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
    cv.imwrite("perim_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    perim_list = pixel_micrometer(perim_list)
    # PLOT
    data = perim_list
    # BOXPLOT
    boxplot = sns.boxplot(y=data, width=0.25)
    # TODO manter frequência de linhas de grade e espaçar frequência de números
    # boxplot.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    # boxplot.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig1 = boxplot.get_figure()
    # plt.title("Crypt axis ratio boxplot")
    # plt.ylabel("Axis ratio")
    fig1.savefig("perim_box_plot.jpg", dpi=300, bbox_inches="tight")
    # plt.show(boxplot)
    plt.clf()
    # DISTRIBUTION
    distribution = sns.distplot(data)  # bin: freedman-diaconis
    # distribution.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # distribution.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # distribution.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # distribution.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig2 = distribution.get_figure()
    plt.title("Perimeter distribution")
    plt.ylabel("Density")
    plt.xlabel("Perimeter (\u03BCm)")
    fig2.savefig("perim_dist_plot.jpg", dpi=300, bbox_inches="tight")
    # plt.show(distribution)
    plt.clf()
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
    cv.imwrite("axisr_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    # PLOT
    data = mama_list
    # BOXPLOT
    boxplot = sns.boxplot(y=data, width=0.25)
    # TODO manter frequência de linhas de grade e espaçar frequência de números
    boxplot.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    boxplot.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig1 = boxplot.get_figure()
    plt.title("Crypt axis ratio boxplot")
    plt.ylabel("Axis ratio")
    fig1.savefig("axis_box_plot.jpg", dpi=300, bbox_inches="tight")
    # plt.show(boxplot)
    plt.clf()
    # DISTRIBUTION
    distribution = sns.distplot(data)  # bin: freedman-diaconis
    distribution.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    distribution.xaxis.set_major_formatter(ticker.ScalarFormatter())
    distribution.yaxis.set_major_locator(ticker.MultipleLocator(1))
    distribution.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig2 = distribution.get_figure()
    plt.title("Crypt axis ratio distribution")
    plt.ylabel("Density")
    plt.xlabel("Axis ratio")
    fig2.savefig("axis_dist_plot.jpg", dpi=300, bbox_inches="tight")
    # plt.show(distribution)
    plt.clf()
    return np.mean(mama_list), np.std(mama_list)


def pixel_micrometer(value_pixel, is_list=True):
    # 51 pixels correspond to 20 micrometers
    PIXEL = 51
    MICROMETER = 20
    if is_list:
        value_micrometer = []
        for value in value_pixel:
            value_micrometer.append((MICROMETER * value) / PIXEL)
        return value_micrometer
    else:
        return (MICROMETER * value_pixel) / PIXEL


def draw_countours(image, crypts_list):
    for crypt in crypts_list:
        cv.drawContours(image, crypt, -1, (0, 0, 255), 3)

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
