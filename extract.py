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
#                     Set a function to call (mosaic; stack; cryptometry)
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
# python extract.py -f mosaic -p midia/main/0000/
# python extract.py -f stack -p midia/main/0000/frame/016-2017EM-PRE/rvss -i 100 200
# python extract.py -f cryptometry -p midia/main/0000/016-2017EM-PRE-0-302TR.tif

import cv2 as cv
import numpy as np
import logendo as le
import subprocess
import sys
from timeit import default_timer as timer


def dir_structure(path, dir_list):
    for dire in dir_list:
        path_dir = path.parents[0] / dire
        if not path_dir.is_dir():
            path_dir.mkdir()
        sub_dir = path_dir / path.stem
        dir_exists(sub_dir)
        sub_dir.mkdir()
        logger.info(
            f'New directory structure was created! Source: {str(sub_dir)}')


def dir_exists(path):
    if path.is_dir():
        logger.info(f'Path {str(path)} already exists. '
                    'Want to send to sandbox? (y/n)')
        option = 'x'
        while (option != 'y' and option != 'n'):
            option = input('*Caution!* To press'
                           ' (n) will overwrite directory\n')
            if option == 'y':
                logger.info('You have pressed (y) option')
                if not ('main' in str(path)):
                    logger.critical("Directory 'main' not found! Exiting...")
                    sys.exit()
                hierarchy = path.parts
                main_index = hierarchy.index('main')
                path_index = len(hierarchy)-(2 + max(0, main_index-1))
                logger.info('Directory was sent to sandbox! Code: '
                            f"{zip_move(path, (path.parents[path_index] / 'sandbox' / hierarchy[main_index+1]))}")
            elif option == 'n':
                logger.info('You have pressed (n) option')
                subprocess.run(f'rm -rf {str(path.resolve())}', shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                logger.info(f'Directory {str(path)} was deleted!')
            else:
                logger.warning('Option unavailable! Press (y) or (n) '
                               'to send (or not) to sandbox')


def is_valid(source):
    import pathlib
    path = pathlib.Path(source)
    if path.is_dir() or path.is_file():
        return True
    return False


def zip_move(path, dest_path):
    start_time = timer()
    logger.info('Initializing zip-move')
    if not dest_path.is_dir():
        dest_path.mkdir()
    count = subprocess.run('find . -maxdepth 1 -type f | wc -l', cwd=dest_path,
                           shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    key_sand = f'{int(count.stdout):04d}'
    subprocess.run(f'zip -r {key_sand}.zip {str(path)}',
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    subprocess.run(f'rm -rf {str(path.resolve())}',
                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    mv = subprocess.run(f'mv -vn {key_sand}.zip {str(dest_path.resolve())}',
                        shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if (mv.stdout == ''):
        logger.error('Destination path already exists!')
    logger.info(f'Finished zip-move. Destiny path: {dest_path}')
    end_time = timer()
    logger.debug(f'Zip-Move function time elapsed: {end_time-start_time:.2f}s')
    return key_sand


def mosaic(source, imagej='/opt/Fiji.app/ImageJ-linux64', extension='mp4'):
    start_time = timer()
    logger.info('Initializing mosaic')
    output = subprocess.run(f'find {source} -type f -name *{extension}', shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    files = output.stdout.splitlines()
    files.sort()
    logger.info(f'No. of {extension} videos found: {len(files)}')
    logger.debug(f'Source: {source} | ImageJ path: {imagej} | Videos: {files}')
    import pathlib
    for index, video_source in enumerate(files):
        start_video = timer()
        logger.info(f'Video {index}: {video_source}')
        path = pathlib.Path(video_source)
        dir_structure(path, ['frame'])
        sub_dir = path.parents[0] / 'frame' / path.stem
        video_frame(video_source, sub_dir)
        rvss_dir = sub_dir / 'rvss'
        rvss_dir.mkdir()
        rvsx_dir = sub_dir / 'rvss-xml'
        rvsx_dir.mkdir()
        if imagej_rvss(imagej, sub_dir, rvss_dir, rvsx_dir):
            stack_frames(rvss_dir, path.stem)
            zip_id = zip_move(rvss_dir, sub_dir)
            logger.info(f'Directory {rvss_dir} was zipped! Code: {zip_id}')
            logger.info(f'Video {video_source} mosaicing completed')
        else:
            logger.warning(
                f'It is not possible to create stack image for video: {video_source}')
            subprocess.run(f'rm -rf {str(rvss_dir.resolve())}',
                           shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        end_video = timer()
        logger.debug(f'Video {index} time elapsed: {start_video-end_video}s')
        subprocess.run(f'mv -vn *tif {str(path.parents[0])}', shell=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    logger.info(f'Finished mosaic. Source: {source}')
    end_time = timer()
    logger.debug(f'Mosaic function time elapsed: {end_time-start_time:.2f}s')


def stack_frames(source, video_id):
    start_time = timer()
    logger.info('Initializing stack frames')
    output = subprocess.run(f"find {source} -type f -name '*tif'", shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    files = output.stdout.splitlines()
    files.sort()
    logger.debug(f'Source: {source} | No. Frames: {len(files)} | '
                 f'Frames: {files}')
    stack = cv.imread(files[0])
    for image_source in files[1:]:
        image = cv.imread(image_source)
        stack = cv.max(stack, image)
    cv.imwrite(f"{video_id}.tif", stack)
    logger.info(f"Finished stack frames {video_id}. Source: {source}")
    end_time = timer()
    logger.debug(
        f'Stack frames function time elapsed: {end_time-start_time:.2f}s')


def substack_frames(source, video_id, interval):
    start_time = timer()
    logger.info('Initializing substack frames')
    logger.debug(f'Frame range to stack: {interval[0]}..{interval[1]}')
    output = subprocess.run(f"find {source} -type f -name '*tif'", shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    files = output.stdout.splitlines()
    files.sort()
    stack = cv.imread(files[interval[0]])
    for image_source in files[interval[0]:interval[1]]:
        image = cv.imread(image_source)
        stack = cv.max(stack, image)
    cv.imwrite(f'{video_id}.tif', stack)
    logger.info(f"Finished substack frames {video_id}. Source: {source}")
    end_time = timer()
    logger.debug(
        f'Substack frames function time elapsed: {end_time-start_time:.2f}s')


def imagej_rvss(imagej, source, output_path, xml, attempt=0):
    start_time = timer()
    logger.info(f'Initializing ImageJ-RVSS wrapper. Attempt: {attempt}')
    imj_cmd = (f"{imagej} --ij2 --headless --console --run rvss.py "
               f"'source=\"{source}/\", output=\"{output_path}/\", xml=\"{xml}/\"'")
    logger.debug(f'ImageJ bash command: {imj_cmd}')
    rvss = subprocess.run(imj_cmd, shell=True,  stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)
    log = rvss.stdout
    logger.debug(f'Attempt: {attempt}. RVSS output: {log}')
    if 'No features model found' in log:
        if attempt < 1:
            begin = log.find('frame')+5
            end = log.find('.png')
            first = int(log[begin:end])
            logger.warning(f"RVSS output: No features model found: frame{first}.png"
                           f"\nRetaking with less frames. Range: 0..{first-1}")
            count = subprocess.run("find . -maxdepth 1 -type f -name '*png' | wc -l", cwd=source,
                                   shell=True,  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            last = int(count.stdout)
            logger.debug(f'Removing frames in the range {first}..{last-1}')
            for index in range(first, last):
                rm_cmd = f"rm -rf {str(source.resolve())}/frame{index}.png"
                subprocess.run(rm_cmd, shell=True,  stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
            subprocess.run(f"rm -rf {str(output_path)}/", shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            output_path.mkdir()
            return imagej_rvss(imagej, source, output_path, xml, attempt+1)
        else:
            logger.error(
                f'RVSS attempt: No features model found. Source: {source}')
            return False
    logger.info(f"Finished ImageJ-RVSS wrapper. Ouput path: {output_path}")
    end_time = timer()
    logger.debug(
        f'ImageJ-RVSS wrapper function time elapsed: {end_time-start_time:.2f}s')
    return True


def video_frame(source, output_path):
    # Convert a video to frame images
    start_time = timer()
    logger.info('Initializing video to frames')
    vidcap = cv.VideoCapture(source)
    success, image = vidcap.read()
    count = 0
    while success:
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = remove_text(gray_frame)
        cv.imwrite(f'{str(output_path)}/frame{count:03d}.png', image)
        success, image = vidcap.read()
        count += 1
    logger.info(f'Finished video to frames. No. frames: {count}')
    end_time = timer()
    logger.debug(
        f'Video to frames function elapsed time {end_time-start_time:.2f}')


def remove_text(image):
    # Remove white text from frame images
    cv.rectangle(image, (0, 0), (80, 30), (0, 0, 0), -1)
    cv.rectangle(image, (496, 504), (576, 584), (0, 0, 0), -1)
    return image


def kmeans_seg(image, k=4):
    start_time = timer()
    logger.info('Initializing k-means')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    blur = cv.GaussianBlur(equalized, (7, 7), 0)

    vectorized = blur.reshape(-1, 1)
    vectorized = np.float32(vectorized)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    logger.debug(f'No. groups: {k} | Criteria: {criteria}')
    ret, label, center = cv.kmeans(vectorized, k, None,
                                   criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = label.reshape((gray.shape))

    segmented = np.zeros(gray.shape, np.uint8)
    segmented[labels == 3] = gray[labels == 3]
    segmented[labels == 2] = gray[labels == 2]
    logger.info('Finished k-means')
    end_time = timer()
    logger.debug(
        f'K-means function time elapsed: {end_time-start_time:.2f}s')
    return segmented


def ellipse_seg(image, iterat=9):
    start_time = timer()
    logger.info('Initializing ellipse segmentation')
    segmented = kmeans_seg(image)

    height = int(0.08 * image.shape[0])
    width = int(0.08 * image.shape[1])
    segmented = cv.copyMakeBorder(segmented, top=height, bottom=height, left=width,
                                  right=width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

    _, thresh = cv.threshold(segmented, 1, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    morph_trans_e = cv.erode(thresh, kernel, iterations=iterat)
    contours_list, hierarchy = cv.findContours(
        morph_trans_e, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    logger.debug(f'No. of elements after erosion: {len(contours_list)} | '
                 f'No. iterations: {iterat}')

    crypts = []
    MIN_AREA = 6200
    for countour in contours_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA:
            crypts.append(countour)
    logger.debug('No. of elements after remove contours smaller than'
                 f' {MIN_AREA} pixels: {len(crypts)}')

    figure = np.zeros(thresh.shape, np.uint8)

    for crypt in crypts:
        hull = cv.convexHull(crypt)
        ellipse = cv.fitEllipse(hull)
        cv.ellipse(figure, ellipse, (255), -1)

    morph_trans_d = cv.dilate(figure, kernel, iterations=iterat)
    crypts_list, hierarchy = cv.findContours(
        morph_trans_d, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    logger.debug(f'No. of crypts after fit ellipse: {len(crypts_list)}')

    crypts_resized = []
    MIN_AREA = 25000
    MAX_AREA = 700000
    for countour in crypts_list:
        area = cv.contourArea(countour)
        if area > MIN_AREA and area < MAX_AREA:
            crypts_resized.append(countour)
    logger.debug('No. of crypts after remove elements under range'
                 f' {MIN_AREA} and {MAX_AREA} pixels: {len(crypts_resized)}')

    image_resized = cv.copyMakeBorder(image, top=height, bottom=height, left=width,
                                      right=width, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0])

    logger.info(f'Number of crypts assessed: {len(crypts_resized)}. '
                'Finished ellipse segmentation')
    end_time = timer()
    logger.debug(
        f'Ellipse segmentation function time elapsed: {end_time-start_time:.2f}s')
    return crypts_resized, image_resized


def cryptometry(source, extension='tif'):
    start_time = timer()
    logger.info('Initializing cryptometry')
    output = subprocess.run(f'find {source} -type f -name *{extension}', shell=True,  stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    files = output.stdout.splitlines()
    files.sort()
    logger.info(f'No. of {extension} images found: {len(files)}')
    logger.debug(f'Source: {source} | Mosaic images: {files}')
    dir_list = ["fig", "data"]
    import pathlib
    for index, image_source in enumerate(files):
        start_image = timer()
        logger.info(f'Image {index}: {image_source}')
        path = pathlib.Path(image_source)
        dir_structure(path, dir_list)
        image = cv.imread(image_source)
        crypts_list, roi_image = ellipse_seg(image)
        draw_countours(roi_image, crypts_list)
        axis_ratio(roi_image.copy(), crypts_list)
        perimeter_shape(roi_image.copy(), crypts_list)
        elong_factor(roi_image.copy(), crypts_list)
        # mean_feret = max_feret(roi_image.copy(), crypts_list)
        mean_feret = max_feret(roi_image.copy(), crypts_list, 'H')
        neighbors_list = neighbors(crypts_list, mean_feret)
        # wall_thickness(roi_image.copy(), crypts_list, neighbors_list)
        wall_thickness(roi_image.copy(), crypts_list, neighbors_list, 'H')
        intercrypt_dist(roi_image.copy(), crypts_list, neighbors_list)
        density(roi_image.copy(), crypts_list)
        for sub_dir in dir_list:
            subprocess.run(f"mv -vn *_{sub_dir}* {str(path.parents[0] / sub_dir / path.stem)}", shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        end_image = timer()
        logger.debug(
            f'Image {index} time elapsed: {end_image-start_image:.2f}s')
    logger.info('Finished cryptometry')
    end_time = timer()
    logger.debug(
        f'Cryptometry function time elapsed: {end_time-start_time:.2f}s')


def density(image, crypts_list):
    start_time = timer()
    logger.info('Initializing density analysis')
    crypts_area = 0
    for crypt in crypts_list:
        crypts_area += ellipse_area(crypt)
    logger.debug('Crypts area: '
                 f'{pixel_micro(crypts_area, ((51**2), (20**2)), is_list=False):.2f}\u03BCm^2')

    _, thresh = cv.threshold(image, 1, 255, cv.THRESH_BINARY)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    background_area = np.sum(thresh == 255)
    logger.debug('Background area: '
                 f'{pixel_micro(background_area, ((51**2), (20**2)), is_list=False):.2f}\u03BCm^2')

    density = [crypts_area/background_area]
    to_csv(density,
           ['density', 'Density', '', 'Ratio'])
    logger.info('Finished density')
    end_time = timer()
    logger.debug(
        f'Density function time elapsed: {end_time-start_time: .2f}s | '
        f'Density: {density[0]:.3f}')


def elong_factor(image, crypts_list):
    start_time = timer()
    logger.info('Initializing elongation factor analysis')
    elongation_list = []
    for index, crypt in enumerate(crypts_list):
        major_axis, minor_axis = ellipse_axis(crypt)
        elongation_list.append(major_axis/minor_axis)
        rect = cv.minAreaRect(crypt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (115, 158, 0), 12)
    cv.imwrite('elong_fig.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(elongation_list, ['elong', 'Elongation factor', '', 'Ratio'])
    logger.info('Finished elongation factor')
    end_time = timer()
    logger.debug(f'Elongation factor function time elapsed: {end_time-start_time:.2f}s | '
                 f'Elongation factor mean: {np.mean(elongation_list):.2f} and '
                 f'std: {np.std(elongation_list):.2f} | Elongation factor list: '
                 + str([f'{value:.2f}' for value in elongation_list]))


def neighbors(crypts_list, mean_diameter):
    start_time = timer()
    logger.info('Initializing neighbors definition')
    MAX_DIST = 2.3 * mean_diameter
    logger.debug(f'Maximal distance to be a neighbor: {MAX_DIST:.2f} pixels')
    neighbors_list = [[] for crypt in range(len(crypts_list))]
    center_list = get_center(crypts_list)
    for crypt_index, first_center in enumerate(center_list):
        for neighbor_index, second_center in enumerate(center_list):
            dist = distance(first_center, second_center)
            if dist < MAX_DIST and dist != 0:
                neighbors_list[crypt_index].append((
                    neighbor_index, dist))
        logger.debug(f'Crypt {crypt_index:03d} - No. of neighbors: '
                     f'{len(neighbors_list[crypt_index])} | Neighbors(index, dist): '
                     + str([f'({index}, {value:.2f})' for index, value in neighbors_list[crypt_index]]))
    logger.info('Finished neighbors definition')
    end_time = timer()
    logger.debug(
        f'Neighbors function time elapsed: {end_time-start_time:.2f}s')
    return neighbors_list


def max_feret(image, crypts_list, algorithm='E'):
    start_time = timer()
    logger.info('Initializing maximal feret diameter analysis')
    algorithm_opt = 'Exhaustive search' if algorithm == 'E' else 'Heuristic'
    logger.info(f'{algorithm_opt} algorithm selected')
    feret_diameters = []
    if algorithm == 'E':
        # EXHAUSTIVE SEARCH
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
    cv.imwrite('feret_fig.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 75])
    mean_feret = np.mean(feret_diameters)
    feret_diameters = pixel_micro(feret_diameters)
    to_csv(feret_diameters, ['feret',
                             'Maximal feret diameter', '', 'Diameter (\u03BCm)'])
    logger.info('Finished maximal feret diameter')
    end_time = timer()
    logger.debug(f'Maximal feret diameter function time elapsed: {end_time-start_time:.2f}s | '
                 f'Maximal feret(\u03BCm) mean: {mean_feret:.2f} and std: {np.std(feret_diameters):.2f} | '
                 'Maximal feret diameter(\u03BCm) list: ' + str([f'{value:.2f}' for value in feret_diameters]))
    return mean_feret


def wall_thickness(image, crypts_list, neighbors_list, algorithm='E'):
    start_time = timer()
    logger.info('Initializing wall thickness analysis')
    MAX_DIST = image.shape[0]
    wall_list = [0] * len(crypts_list)
    algorithm_opt = 'Exhaustive search' if algorithm == 'E' else 'Heuristic'
    logger.info(f'{algorithm_opt} algorithm selected')
    for crypt_index, crypt in enumerate(crypts_list):
        if len(neighbors_list[crypt_index]) == 0:
            continue
        min_wall = MAX_DIST
        wall_crypt_point = [0]
        wall_neighbor_point = [0]
        for neighbor in neighbors_list[crypt_index]:
            if algorithm == 'E':
                # EXHAUSTIVE SEARCH
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
    cv.imwrite('wall_fig.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 75])
    wall_list = pixel_micro(wall_list)
    to_csv(wall_list, ['wall',
                       'Wall thickness', '', 'Distance (\u03BCm)'])
    logger.info('Finished wall thickness')
    end_time = timer()
    logger.debug(f'Wall thickness function time elapsed: {end_time-start_time:.2f}s | '
                 f'Wall thickness(\u03BCm) mean: {np.mean(wall_list):.2f} and std: {np.std(wall_list):.2f} | '
                 'Wall thickness(\u03BCm) list: ' + str([f'{value:.2f}' for value in wall_list]))


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


def intercrypt_dist(image, crypts_list, neighbors_list):
    start_time = timer()
    logger.info('Initializing intercrypt distance analysis')
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
    cv.imwrite('dist_fig.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 75])
    intercrypt_list = pixel_micro(intercrypt_list)
    to_csv(intercrypt_list, ['dist',
                             'Mean intercrypt distance', '', 'Distance (\u03BCm)'])
    min_dist_list = pixel_micro(min_dist_list)
    to_csv(min_dist_list, ['min_dist',
                           'Minimal intercrypt distance', '', 'Distance (\u03BCm)'])
    logger.info('Finished intercrypt distance')
    end_time = timer()
    logger.debug(f'Intercrypt distance function time elapsed: {end_time-start_time:.2f}s | '
                 f'Mean intercrypt distance(\u03BCm) mean: {np.mean(intercrypt_list):.2f} and std: {np.std(intercrypt_list):.2f} | '
                 'Mean intercrypt distance(\u03BCm) list: ' + str([f'{value:.2f}' for value in intercrypt_list]))
    logger.debug(f'Minimal intercrypt distance(\u03BCm) mean: {np.mean(min_dist_list):.2f} and std: {np.std(min_dist_list):.2f} | '
                 'Minimal intercrypt distance(\u03BCm) list: ' + str([f'{value:.2f}' for value in min_dist_list]))


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


def perimeter_shape(image, crypts_list):
    start_time = timer()
    logger.info('Initializing perimeter-shape analysis')
    perim_list = []
    spher_list = []
    roundness_list = []
    for index, crypt in enumerate(crypts_list):
        major_axis, minor_axis = ellipse_axis(crypt)
        perimeter = ellipse_perim((major_axis / 2), (minor_axis / 2))
        perimeter = pixel_micro(perimeter, is_list=False)
        perim_list.append(perimeter)
        area = ellipse_area(crypt)
        area = pixel_micro(area, ((51**2), (20**2)), is_list=False)
        spher = (4 * np.pi * area) / (perimeter ** 2)*100
        major_axis = pixel_micro(major_axis, is_list=False)
        roundness = (4*(area/(np.pi * (major_axis ** 2))))*100
        spher_list.append(spher)
        roundness_list.append(roundness)
        logger.debug(f'Crypt {index:03d} - (2a: {major_axis:.2f}, '
                     f'2b: {pixel_micro(minor_axis, is_list=False):.2f})\u03BCm | '
                     f'Perimeter: {perimeter:.2f}\u03BCm | Area: {area:.2f}\u03BCm^2 | '
                     f'Sphericity: {spher:.2f}% | Roundness: {roundness:.2f}%')
    cv.imwrite('perim_fig.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(perim_list, ['perim', 'Crypts Perimeter',
                        '', 'Perimeter (\u03BCm)'])
    to_csv(spher_list, ['spher', 'Crypts sphericity', '', 'Sphericity (%)'])
    to_csv(roundness_list, ['round', 'Crypts roundness', '', 'Roundness (%)'])
    logger.info('Finished perimeter-shape')
    end_time = timer()
    logger.debug(f'Perimeter-shape function time elapsed: {end_time-start_time:.2f}s | '
                 f'Perimeter(\u03BCm) mean: {np.mean(perim_list):.2f} and '
                 f'std: {np.std(perim_list):.2f} | Sphericity mean: {np.mean(spher_list):.2f} '
                 f'and std: {np.std(spher_list):.2f} | Roundness mean: {np.mean(roundness_list):.2f} '
                 f'and std: {np.std(roundness_list):.2f} ')


def ellipse_area(crypt):
    major_axis, minor_axis = ellipse_axis(crypt)
    return np.pi * (major_axis / 2) * (minor_axis / 2)


def ellipse_axis(crypt):
    rect = cv.minAreaRect(crypt)
    (x, y), (width, height), angle = rect
    return max(width, height), min(width, height)


def ellipse_perim(a, b, n=10):
    h = ((a-b)**2)/((a+b)**2)
    summation = 0
    import math
    for i in range(n):
        summation += ((math.gamma(0.5+1)/(math.factorial(i) *
                                          math.gamma((0.5+1)-i)))**2)*(np.power(h, i))
    return np.pi * (a+b) * summation


def axis_ratio(image, crypts_list):
    # Ratio between major and minor axis (Ma/ma ratio)
    start_time = timer()
    logger.info('Initializing axis ratio analysis')
    axisr_list = []
    for index, crypt in enumerate(crypts_list):
        x, y, width, height = cv.boundingRect(crypt)
        logger.debug(f'Crypt {index:03d} - (W: {width:.2f},'
                     f' H: {height:.2f}) pixels')
        axisr_list.append(max(width, height) / min(width, height))
        cv.rectangle(image, (x, y), (x + width, y + height), (115, 158, 0), 12)
    cv.imwrite('axisr_fig.jpg', image, [cv.IMWRITE_JPEG_QUALITY, 75])
    to_csv(axisr_list, ['axisr', 'Axis Ratio', '', 'Ratio'])
    logger.info('Finished axis ratio')
    end_time = timer()
    logger.debug(f'Axis ratio function time elapsed: {end_time-start_time:.2f}s | '
                 f'Axis ratio mean: {np.mean(axisr_list):.2f} and std: {np.std(axisr_list):.2f} | '
                 'Axis ratio list: ' + str([f'{value:.2f}' for value in axisr_list]))


def pixel_micro(value_pixel, ratio=(51, 20), is_list=True):
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


def to_csv(data, labels, is_list=False):
    import csv
    with open(f"{labels[0]}_data.csv", mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(labels)
        if not is_list:
            writer.writerow(data)
        else:
            for row in data:
                writer.writerow(row)


def draw_countours(image, crypts_list):
    for crypt in crypts_list:
        cv.drawContours(image, crypt, -1, (115, 158, 0), 12)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--function", type=str, required=True,
                    help="Set a function to call (video_frame; video_frame_crop, cryptometry)")
    ap.add_argument("-v", "--verbose", help="Increase output verbosity",
                    action="store_true")
    ap.add_argument("-p", "--path", type=str, required=False,
                    help="Input file or directory of images path")
    ap.add_argument("-i", "--interval", nargs='+', type=int, required=False,
                    help="Define range of frames in Stack function")

    args = vars(ap.parse_args())
    function = args["function"]
    source = args["path"]
    interval = args["interval"]
    verbose = args["verbose"]

    global logger
    if verbose:
        logger = le.logging.getLogger('debug')

    if is_valid(source):
        logger.info('\n\nFRAMEWORK FOR ENDOMICROSCOPY ANALYSIS\n')
        if (function == "mosaic"):
            mosaic(source)
        elif (function == "stack"):
            name = f'stack-{interval[0]}-{interval[1]}'
            substack_frames(source, name, interval)
        elif (function == "cryptometry"):
            cryptometry(source)
        else:
            logger.error("Undefined function")
    else:
        logger.error(
            f'The path "{source}" is not a valid source! Exiting...')


if __name__ == "__main__":
    le.setup_logging()
    logger = le.logging.getLogger('default')
    main()
