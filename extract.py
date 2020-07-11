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
#       VERSION:  0.3
#       CREATED:  14/02/2020
#      REVISION:  ---
# =============================================================================

# USAGE
# python extract.py -f video_frame -p midia/main/1234/
# python extract.py -f cryptometry -p midia/main/1234/016-2017EM-PRE-0-302TR.tif

import cv2 as cv
import numpy as np
import subprocess
import sys


def dir_structure(path, dir_list):
    if not path.is_file():
        print(f"The path {str(path)} is not a valid file name! Exiting...")
        sys.exit()
    for dire in dir_list:
        path_dir = path.parents[0] / dire
        if not path_dir.is_dir():
            path_dir.mkdir()
        sub_dir = path_dir / path.stem
        dir_exists(sub_dir)
        sub_dir.mkdir()
        print(f"New directory structure was created! Source: {str(sub_dir)}")


def dir_exists(path):
    if path.is_dir():
        option = input(
            f" Path {str(path)} already exists! Want to send to sandbox? (y/n)"
            " *Caution!* To press n will overwrite directory\n")
        if option == "y":
            if not ("main" in str(path)):
                print("Directory 'main' not found! Exiting...")
                sys.exit()
            hierarchy = path.parts
            main_index = hierarchy.index("main")
            path_index = len(hierarchy)-(2 + max(0, main_index-1))
            print("Directory was sent to sandbox! Code: "
                  f"{send_sandbox(path, (path.parents[path_index] / 'sandbox' / hierarchy[main_index+1]))}")
        elif option == "n":
            subprocess.run(f"rm -rf {str(path.resolve())}", shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            print(f"Directory {str(path)} was deleted!")
        else:
            print("Option unavailable! Exiting...")
            sys.exit()


def send_sandbox(path, dest_path):
    if not dest_path.is_dir():
        dest_path.mkdir()
    count = subprocess.run("find . -maxdepth 1 -type f | wc -l", cwd=dest_path,
                           shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    key_sand = f"{int(count.stdout):04d}"
    subprocess.run(f"zip -r {key_sand}.zip {str(path)}",
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    subprocess.run(f"rm -rf {str(path.resolve())}",
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    mv = subprocess.run(f"mv -vn {key_sand}.zip {str(dest_path.resolve())}",
                        shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if (mv.stdout == ''):
        print("Error to move: destination path already exists!")
    return key_sand


def video_frame(source):
    # Convert a video to frame images
    files = subprocess.run(f"find {source} -type f -name *mp4", shell=True,  stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
    import pathlib
    for video_source in files.stdout.splitlines():
        path = pathlib.Path(video_source)
        dir_structure(path, ["frame"])
        sub_dir = path.parents[0] / "frame" / path.stem
        vidcap = cv.VideoCapture(video_source)
        success, image = vidcap.read()
        count = 0
        while success:
            gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = remove_text(gray_frame)
            cv.imwrite(f"{str(sub_dir)}/frame{count:03d}.png", image)
            success, image = vidcap.read()
            count += 1
        print(f"Finished: {video_source}")


def remove_text(image):
    # Remove white text from frame images
    cv.rectangle(image, (0, 0), (80, 30), (0, 0, 0), -1)
    cv.rectangle(image, (496, 504), (576, 584), (0, 0, 0), -1)
    return image


def full_processing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    contours_list, hierarchy = cv.findContours(
        thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    lista = []
    MIN_AREA = 9000
    MAX_AREA = 145000
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            lista.append(countour)

    figure = np.zeros(image.shape[:-1], np.uint8)
    draw_object(figure, lista)  # 1 All

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    morph_trans = cv.erode(thresh.copy(), kernel, iterations=17)

    contours_list, hierarchy = cv.findContours(
        morph_trans, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    lista = []
    MIN_AREA = 9000
    MAX_AREA = 145000
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            lista.append(countour)

    draw_object(figure, lista)  # 2 Edge

    morph_trans = cv.dilate(thresh.copy(), kernel, iterations=5)

    contours_list, hierarchy = cv.findContours(
        morph_trans, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    lista = []
    MIN_AREA = 9000
    MAX_AREA = 145000
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            lista.append(countour)

    draw_object(figure, lista)  # 3 Center

    erode = cv.erode(figure.copy(), kernel, iterations=14)
    contours_list, hierarchy = cv.findContours(
        erode, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    lista = []
    MIN_AREA = 5000
    MAX_AREA = 145000
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            lista.append(countour)

    separed = np.zeros(image.shape[:-1], np.uint8)
    draw_object(separed, lista)  # 4 Separed crypts

    final = np.zeros(image.shape[:-1], np.uint8)

    for crypt in lista:
        hull = cv.convexHull(crypt)
        ellipseB = cv.fitEllipse(hull)
        cv.ellipse(final, ellipseB, (255), -1)

    contours_list, hierarchy = cv.findContours(
        final, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 5 Estimated ellipses

    print(f"Number of crypts assessed: {len(contours_list)}")
    return contours_list


def draw_object(image, crypts_list):
    for crypt in crypts_list:
        cv.drawContours(image, [crypt], -1, (255), -1)


def low_processing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    dilate = cv.dilate(thresh, kernel, iterations=10)
    morph_trans = cv.erode(dilate, kernel, iterations=15)
    processed_image = morph_trans
    return processed_image


def segmentation(image):
    processed_image = low_processing(image)
    contours_list, hierarchy = cv.findContours(
        processed_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    crypts_list = []
    MIN_AREA = 9000
    MAX_AREA = 145000
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            crypts_list.append(countour)
    print(f"Number of crypts assessed: {len(crypts_list)}")
    return crypts_list


def cryptometry(source):
    import pathlib
    path = pathlib.Path(source)
    dir_list = ["fig", "plot", "data"]
    dir_structure(path, dir_list)
    print("Initialize cryptometry")
    image = cv.imread(source)
    from timeit import default_timer as timer
    print("Measures\t\t\t\t TIME(s)")
    start = timer()
    crypts_list = full_processing(image.copy())
    draw_countours(image, crypts_list)
    end = timer()
    print(f"Segmentation and draw\t\t\t {end-start:.2f}")
    start = timer()
    axis_ratio(image.copy(), crypts_list)
    end = timer()
    print(f"Axis ratio\t\t\t\t {end-start:.2f}")
    start = timer()
    perimeter(image.copy(), crypts_list)
    end = timer()
    print(f"Perimeter and Sphericity\t\t {end-start:.2f}")
    start = timer()
    intercrypt_distance(image.copy(), crypts_list)
    end = timer()
    print(f"Mean and Minimal intercrypt distance\t {end-start:.2f}")
    start = timer()
    # wall_thickness(image.copy(), crypts_list)
    wall_thickness(image.copy(), crypts_list, 'H')
    end = timer()
    print(f"Wall Thickness\t\t\t\t {end-start:.2f}")
    start = timer()
    # maximal_feret(image.copy(), crypts_list)
    maximal_feret(image.copy(), crypts_list, 'H')
    end = timer()
    print(f"Max Feret\t\t\t\t {end-start:.2f}")
    start = timer()
    elongation_factor(image.copy(), crypts_list)
    end = timer()
    print(f"Elongation factor\t\t\t {end-start:.2f}")
    start = timer()
    roundness(crypts_list)
    end = timer()
    print(f"Roundness\t\t\t\t {end-start:.2f}")
    start = timer()
    density(image.copy(), crypts_list)
    end = timer()
    print(f"Density\t\t\t\t\t {end-start:.2f} \nFinished cryptometry")
    for sub_dir in dir_list:
        subprocess.run(f"mv -vn *_{sub_dir}* {str(path.parents[0] / sub_dir / path.stem)}", shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)


def density(image, crypts_list):
    crypts_img = np.zeros(image.shape[:-1], np.uint8)
    for crypt in crypts_list:
        cv.drawContours(crypts_img, [crypt], -1, (255),
                        thickness=-1)
    crypts_area = np.sum(crypts_img == 255)
    processed_image = low_processing(image)
    height = int(0.01 * processed_image.shape[0])
    width = int(0.01 * processed_image.shape[1])
    border = cv.copyMakeBorder(processed_image, top=height, bottom=height,
                               left=width, right=width,
                               borderType=cv.BORDER_CONSTANT, value=[255, 255,
                                                                     255])
    contours_list, hierarchy_list = cv.findContours(
        border, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    background_img = np.zeros(border.shape, np.uint8)
    for index, hierarchy in enumerate(hierarchy_list[0]):
        if hierarchy[0] == -1 and hierarchy[3] == 0:
            cv.drawContours(background_img, [contours_list[index]], -1, (255),
                            thickness=-1)
            break
    background_area = np.sum(background_img == 255)
    # print(crypts_area/background_area)  # >>> Criar CSV
    return crypts_area/background_area


def roundness(crypts_list):
    roundness_list = []
    angle_list = []
    for crypt in crypts_list:
        area = cv.contourArea(crypt)
        rect = cv.minAreaRect(crypt)
        (x, y), (width, height), angle = rect
        angle_list.append(angle)
        major_axis = max(width, height)
        roundness_list.append((4*(area/(np.pi * (major_axis ** 2))))*100)
    to_csv(angle_list, ["angle", "Crypts angles", "", "Angles (degrees)"])
    to_csv(roundness_list, ["round", "Crypts roundness", "", "Roundness (%)"])


def elongation_factor(image, crypts_list):
    elongation_list = []
    for crypt in crypts_list:
        rect = cv.minAreaRect(crypt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        (x, y), (width, height), angle = rect
        elongation_list.append(max(width, height)/min(width, height))
        cv.drawContours(image, [box], 0, (0, 0, 255), 3)
    cv.imwrite("elong_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(elongation_list, ["elong", "Elongation factor", "", "Ratio"])


def neighbors(crypts_list):
    MAX_DIST = 700
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
                for pointB in crypt[index+1:]:
                    dist = distance(pointA[0], pointB[0])
                    if dist > max_dist:
                        max_dist = dist
                        max_pointA[0] = pointA[0]
                        max_pointB[0] = pointB[0]
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
                cv.circle(image, left, 7, (255, 0, 0), -1)
                cv.circle(image, right, 7, (255, 0, 0), -1)
                cv.line(image, left, right, (255, 0, 0), thickness=3)
                feret_diameters.append(x_distance)
            else:
                cv.circle(image, top, 7, (255, 0, 0), -1)
                cv.circle(image, bottom, 7, (255, 0, 0), -1)
                cv.line(image, top, bottom, (255, 0, 0), thickness=3)
                feret_diameters.append(y_distance)
    cv.imwrite("feret_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    feret_diameters = pixel_micrometer(feret_diameters)
    to_csv(feret_diameters, ["feret",
                             "Maximal feret diameter", "", "Diameter (\u03BCm)"])


def wall_thickness(image, crypts_list, algorithm='B'):
    MAX_DIST = 700
    wall_list = [0] * len(crypts_list)
    neighbors_list = neighbors(crypts_list)
    for crypt_index, crypt in enumerate(crypts_list):
        if len(neighbors_list[crypt_index]) == 0:
            continue
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
                         (neighbor_center[0] - crypt_center[0])) if (neighbor_center[0] - crypt_center[0]) != 0 else 0
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
    to_csv(wall_list, ["wall",
                       "Wall thickness", "", "Distance (\u03BCm)"])


def between_points(slope, first_point, second_point, mid_point):
    import math
    left = distance(first_point, mid_point)
    right = distance(mid_point, second_point)
    link = distance(first_point, second_point)
    return math.isclose(left+right, link, abs_tol=abs(slope))


def collinear(slope, first_point, collinear_point):
    import math
    equation = ((slope*collinear_point[0]) -
                (slope*first_point[0]))+first_point[1]
    return math.isclose(equation, collinear_point[1], abs_tol=(abs(slope))+1)


def intercrypt_distance(image, crypts_list):
    MAX_DIST = 700
    center_list = get_center(crypts_list)
    for center in center_list:
        cv.circle(image,  (center), 7, (0, 0, 255), -1)
    intercrypt_list = []
    min_dist_list = []
    neighbors_list = neighbors(crypts_list)
    for index, first_center in enumerate(center_list):
        min_dist = MAX_DIST
        for neighbor in neighbors_list[index]:
            second_center = center_list[neighbor[0]]
            cv.line(image, first_center, second_center,
                    (0, 0, 255), thickness=3)
            intercrypt_list.append(neighbor[1])
            if neighbor[1] < min_dist:
                min_dist = neighbor[1]
        min_dist_list.append(min_dist)
    cv.imwrite("dist_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    intercrypt_list = pixel_micrometer(intercrypt_list)
    to_csv(intercrypt_list, ["dist",
                             "Intercrypt distance", "", "Distance (\u03BCm)"])
    min_dist_list = pixel_micrometer(min_dist_list)
    to_csv(min_dist_list, ["min_dist",
                           "Minimal intercrypt distance", "", "Distance (\u03BCm)"])


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
        spher_list.append((4 * np.pi * area) / (perimeter ** 2)*100)
    cv.imwrite("perim_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    perim_list = pixel_micrometer(perim_list)
    to_csv(perim_list, ["perim", "Crypts Perimeter",
                        "", "Perimeter (\u03BCm)"])
    to_csv(spher_list, ["spher", "Crypts sphericity", "", "Sphericity (%)"])


def axis_ratio(image, crypts_list):
    # Ratio between major and minor axis (Ma/ma ratio)
    # Give the mean and standard deviation of the ratio between the width and
    # the heigth of the box containing the crypt
    mama_list = []
    for crypt in crypts_list:
        x, y, width, heigth = cv.boundingRect(crypt)
        mama_list.append(max(width, heigth) / min(width, heigth))
        cv.rectangle(image, (x, y), (x + width, y + heigth), (0, 0, 255), 3)
    cv.imwrite("axisr_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(mama_list, ["axisr", "Axis Ratio", "", "Ratio"])


def pixel_micrometer(value_pixel, ratio=(51, 20), is_list=True):
    # 51 pixels correspond to 20 micrometers by default
    PIXEL = ratio[0]
    MICROMETER = ratio[1]
    if is_list:
        value_micrometer = []
        for value in value_pixel:
            value_micrometer.append((MICROMETER * int(value)) / PIXEL)
        return value_micrometer
    else:
        return (MICROMETER * value_pixel) / PIXEL


def to_csv(data, labels):
    import csv
    with open(f"{labels[0]}_data.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(labels)
        writer.writerow(data)


def draw_countours(image, crypts_list):
    for crypt in crypts_list:
        cv.drawContours(image, crypt, -1, (0, 0, 255), 6)


def main():
    import argparse
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
    elif (function == "cryptometry"):
        cryptometry(source)
    else:
        print("Undefined function")


if __name__ == "__main__":
    import timeit
    repetitions = 1
    print(
        f"Mean time elapsed {timeit.timeit(main, number=repetitions)/repetitions:.2f}s"
        f" in {repetitions} executions")
