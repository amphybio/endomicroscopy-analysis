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
#       VERSION:  0.5
#       CREATED:  14/02/2020
#      REVISION:  ---
# =============================================================================

# USAGE
# python extract.py -f video-frame -p midia/main/0000/
# python extract.py -f cryptometry -p midia/main/0000/016-2017EM-PRE-0-302TR.tif

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


def kmeans_seg(image, k=4):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    blur = cv.GaussianBlur(equalized, (7, 7), 0)

    vectorized = blur.reshape(-1, 1)
    vectorized = np.float32(vectorized)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    ret, label, center = cv.kmeans(vectorized, k, None,
                                   criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = label.reshape((gray.shape))

    segmented = np.zeros(gray.shape, np.uint8)
    segmented[labels == 3] = gray[labels == 3]
    segmented[labels == 2] = gray[labels == 2]
    return segmented


def ellipse_seg(image):
    segmented = kmeans_seg(image)

    height = int(0.08 * image.shape[0])
    width = int(0.08 * image.shape[1])
    segmented = cv.copyMakeBorder(segmented, top=height, bottom=height, left=width,
                                  right=width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

    _, thresh = cv.threshold(segmented, 1, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    morph_trans_e = cv.erode(thresh, kernel, iterations=9)
    contours_list, hierarchy = cv.findContours(
        morph_trans_e, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    crypts = []
    MIN_AREA = 6200
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA:
            crypts.append(countour)

    figure = np.zeros(thresh.shape, np.uint8)

    for crypt in crypts:
        hull = cv.convexHull(crypt)
        ellipse = cv.fitEllipse(hull)
        cv.ellipse(figure, ellipse, (255), -1)

    morph_trans_d = cv.dilate(figure, kernel, iterations=9)
    crypts_list, hierarchy = cv.findContours(
        morph_trans_d, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    crypts_resized = []
    MIN_AREA = 25000
    MAX_AREA = 700000
    for countour in crypts_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            crypts_resized.append(countour)

    image_resized = cv.copyMakeBorder(image, top=height, bottom=height, left=width,
                                      right=width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

    print(f"Number of crypts assessed: {len(crypts_resized)}")
    return crypts_resized, image_resized


def cryptometry(source):
    import pathlib
    path = pathlib.Path(source)
    dir_list = ["fig", "data"]
    dir_structure(path, dir_list)
    print("Initialize cryptometry")
    image_source = cv.imread(source)
    from timeit import default_timer as timer
    print("Measures\t\t\t\t TIME(s)")
    start = timer()
    crypts_list, image = ellipse_seg(image_source)
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
    elongation_factor(image.copy(), crypts_list)
    end = timer()
    print(f"Elongation factor\t\t\t {end-start:.2f}")
    start = timer()
    roundness(crypts_list)
    end = timer()
    print(f"Roundness\t\t\t\t {end-start:.2f}")
    start = timer()
    # mean_feret = maximal_feret(image.copy(), crypts_list)
    mean_feret = maximal_feret(image.copy(), crypts_list, 'H')
    end = timer()
    print(f"Max Feret\t\t\t\t {end-start:.2f}")
    start = timer()
    neighbors_list = neighbors(crypts_list, mean_feret)
    # wall_thickness(image.copy(), crypts_list, neighbors_list)
    wall_thickness(image.copy(), crypts_list, neighbors_list, 'H')
    end = timer()
    print(f"Wall Thickness\t\t\t\t {end-start:.2f}")
    start = timer()
    intercrypt_distance(image.copy(), crypts_list, neighbors_list)
    end = timer()
    print(f"Mean and Minimal intercrypt distance\t {end-start:.2f}")
    start = timer()
    density(image_source.copy(), crypts_list)
    end = timer()
    print(f"Density\t\t\t\t\t {end-start:.2f} \nFinished cryptometry")

    for sub_dir in dir_list:
        subprocess.run(f"mv -vn *_{sub_dir}* {str(path.parents[0] / sub_dir / path.stem)}", shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)


def density(image, crypts_list):
    crypts_area = 0
    for crypt in crypts_list:
        crypts_area += ellipse_area(crypt)

    _, thresh = cv.threshold(image, 1, 255, cv.THRESH_BINARY)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    background_area = np.sum(thresh == 255)

    density = [crypts_area/background_area]
    to_csv(density,
           ["density", "Density", "", "Ratio"])


def roundness(crypts_list):
    roundness_list = []
    for crypt in crypts_list:
        major_axis, _ = ellipse_axis(crypt)
        area = ellipse_area(crypt)
        roundness_list.append((4*(area/(np.pi * (major_axis ** 2))))*100)
    to_csv(roundness_list, ["round", "Crypts roundness", "", "Roundness (%)"])


def elongation_factor(image, crypts_list):
    elongation_list = []
    for crypt in crypts_list:
        major_axis, minor_axis = ellipse_axis(crypt)
        elongation_list.append(major_axis/minor_axis)
        rect = cv.minAreaRect(crypt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (115, 158, 0), 12)
    cv.imwrite("elong_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(elongation_list, ["elong", "Elongation factor", "", "Ratio"])


def neighbors(crypts_list, mean_diameter):
    MAX_DIST = 2.3 * mean_diameter
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
            cv.circle(image, tuple(max_pointA[0]), 7, (115, 158, 0), -1)
            cv.circle(image, tuple(max_pointB[0]), 7, (115, 158, 0), -1)
            cv.line(image, tuple(max_pointA[0]), tuple(
                max_pointB[0]), (115, 158, 0), thickness=12)
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
                cv.line(image, left, right, (255, 0, 0), thickness=12)
                feret_diameters.append(x_distance)
            else:
                cv.circle(image, top, 7, (255, 0, 0), -1)
                cv.circle(image, bottom, 7, (255, 0, 0), -1)
                cv.line(image, top, bottom, (255, 0, 0), thickness=12)
                feret_diameters.append(y_distance)
    cv.imwrite("feret_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    mean_feret = np.mean(feret_diameters)
    feret_diameters = pixel_micrometer(feret_diameters)
    to_csv(feret_diameters, ["feret",
                             "Maximal feret diameter", "", "Diameter (\u03BCm)"])
    return mean_feret


def wall_thickness(image, crypts_list, neighbors_list, algorithm='B'):
    MAX_DIST = image.shape[0]
    wall_list = [0] * len(crypts_list)
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
        cv.circle(image,  tuple(wall_crypt_point[0]), 7, (115, 158, 0), -1)
        cv.circle(image,  tuple(wall_neighbor_point[0]), 7, (115, 158, 0), -1)
        cv.line(image, tuple(wall_crypt_point[0]), tuple(
            wall_neighbor_point[0]), (115, 158, 0), 12)
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


def intercrypt_distance(image, crypts_list, neighbors_list):
    MAX_DIST = image.shape[0]
    center_list = get_center(crypts_list)
    for center in center_list:
        cv.circle(image,  (center), 7, (115, 158, 0), -1)
    intercrypt_list = []
    min_dist_list = []
    for index, first_center in enumerate(center_list):
        min_dist = MAX_DIST
        for neighbor in neighbors_list[index]:
            second_center = center_list[neighbor[0]]
            cv.line(image, first_center, second_center,
                    (115, 158, 0), thickness=12)
            intercrypt_list.append(neighbor[1])
            if neighbor[1] < min_dist:
                min_dist = neighbor[1]
        min_dist_list.append(min_dist)
    cv.imwrite("dist_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    intercrypt_list = pixel_micrometer(intercrypt_list)
    to_csv(intercrypt_list, ["dist",
                             "Mean intercrypt distance", "", "Distance (\u03BCm)"])
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
        major_axis, minor_axis = ellipse_axis(crypt)
        perimeter = ellipse_perimeter((major_axis / 2), (minor_axis / 2))
        perim_list.append(perimeter)
        area = ellipse_area(crypt)
        spher_list.append((4 * np.pi * area) / (perimeter ** 2)*100)
    cv.imwrite("perim_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    perim_list = pixel_micrometer(perim_list)
    to_csv(perim_list, ["perim", "Crypts Perimeter",
                        "", "Perimeter (\u03BCm)"])
    to_csv(spher_list, ["spher", "Crypts sphericity", "", "Sphericity (%)"])


def ellipse_area(crypt):
    major_axis, minor_axis = ellipse_axis(crypt)
    return np.pi * (major_axis / 2) * (minor_axis / 2)


def ellipse_axis(crypt):
    rect = cv.minAreaRect(crypt)
    (x, y), (width, height), angle = rect
    return max(width, height), min(width, height)


def ellipse_perimeter(a, b, n=10):
    h = ((a-b)**2)/((a+b)**2)
    summation = 0
    import math
    for i in range(n):
        summation += ((math.gamma(0.5+1)/(math.factorial(i) *
                                          math.gamma((0.5+1)-i)))**2)*(np.power(h, i))
    return np.pi * (a+b) * summation


def axis_ratio(image, crypts_list):
    # Ratio between major and minor axis (Ma/ma ratio)
    # Give the mean and standard deviation of the ratio between the width and
    # the heigth of the box containing the crypt
    axisr_list = []
    for crypt in crypts_list:
        x, y, width, heigth = cv.boundingRect(crypt)
        axisr_list.append(max(width, heigth) / min(width, heigth))
        cv.rectangle(image, (x, y), (x + width, y + heigth), (115, 158, 0), 12)
    cv.imwrite("axisr_fig.jpg", image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(axisr_list, ["axisr", "Axis Ratio", "", "Ratio"])


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
        cv.drawContours(image, crypt, -1, (115, 158, 0), 12)


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
    if (function == "video-frame"):
        video_frame(source)
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
