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
# python extract.py --path midia/016-2017
# python extract.py --path midia/tres

import cv2 as cv
import numpy as np
import argparse
import os
import sys


def video_to_frame(source):
    vidcap = cv.VideoCapture(source)
    success, image = vidcap.read()
    count = 0
    path = os.path.splitext(source)[0]
    os.mkdir(path)
    while success:
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imwrite(path+"/frame%03d.png" % count, gray_frame)
        success, image = vidcap.read()
        count += 1
    file = os.path.basename(source)
    print('Finished ', file)


def video_to_frame_cropped(source):
    vidcap = cv.VideoCapture(source)
    success, image = vidcap.read()
    count = 0
    path = os.path.splitext(source)[0]
    os.mkdir(path)
    while success:
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        crop_img = gray_frame[75:500, 75:500]
        cv.imwrite(path+"/frame%03d.png" % count, crop_img)
        success, image = vidcap.read()
        count += 1
    file = os.path.basename(source)
    print('Finished ', file)


def stich_stack(source):
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


def stich_stack_loop(source):
    list_images = sorted(os.listdir(source))
    images = []
    for image_name in list_images:
        image = cv.imread(source+'/'+image_name)
        images.append(image)
    two_images = []
    two_images.append(images.pop(0))
    for image in range(0, len(images)):
        two_images.append(images.pop(0))
        stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)
        status, pano = stitcher.stitch(two_images)
        if status != cv.Stitcher_OK:
            print("Can't stitch images, error code = %d" % status, image)
            sys.exit(-1)
        two_images = []
        two_images.append(pano)
    cv.imwrite("teste.png", two_images.pop(0))
    print("stitching completed successfully.")
    print(image)


def stich_loop(source):
    list_images = sorted(os.listdir(source))
    images = []
    for image_name in list_images:
        image = cv.imread(source+'/'+image_name)
        images.append(image)
    two_images = []
    two_images.append(images.pop(0))
    for image in range(0, len(images)):
        two_images.append(images.pop(0))
        result = stitch(two_images)
        if result is None:
            print("Can't stitch images", image)
            sys.exit(-1)
        two_images = []
        two_images.append(result)
    cv.imwrite("teste.png", two_images.pop(0))
    print("stitching completed successfully.")
    print(image)


def stitch(images, ratio=0.75, reproj_thresh=4.0):
    (imageB, imageA) = images
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None
    (matches, H, status) = M
    result = cv.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result


def detectAndDescribe(image):
    descriptor = cv.xfeatures2d.SIFT_create(0, 5, 0.04, 10, 1.6)
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    matcher = cv.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, reprojThresh)
        return (matches, H, status)
    return None


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, required=True, help="path to input directory of images")
args = vars(ap.parse_args())

source = args["path"]

video_to_frame(source)
# video_to_frame_cropped(source)
# stich_stack(source)
# stich_loop(source) # Esse não funciona
# stich_stack_loop(source)
