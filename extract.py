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

#===============================================================================
#       ARQUIVO:  extract.py
#
#     DESCRIÇÃO: Script to extract features of endomicroscopy images
#
#        OPÇÕES:  ---
#    REQUISITOS:  OpenCV, Python
#          BUGS:  ---
#         NOTAS:  ---
#         AUTOR:  Alan U. Sabino <alan.sabino@usp.br>
#        VERSÃO:  0.1
#       CRIAÇÃO:  14/02/2020
#       REVISÃO:  ---
#===============================================================================

import cv2 as cv
import numpy as np
import os

def video_to_frame(source):
  vidcap = cv.VideoCapture(source)
  success,image = vidcap.read()
  count = 0
  dir = os.path.splitext(source)[0]
  os.mkdir(dir)
  while success:
    gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(dir+"/frame%d.png" % count, gray_frame)
    success,image = vidcap.read()
    count += 1
  file = os.path.basename(source)
  print('Finished ', file)

def video_to_frame_cropped(source):
  vidcap = cv.VideoCapture(source)
  success,image = vidcap.read()
  count = 0
  dir = os.path.splitext(source)[0]
  os.mkdir(dir)
  while success:
    gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    crop_img = gray_frame[83:498, 83:498]
    cv.imwrite(dir+"/frame%d.png" % count, crop_img)
    success,image = vidcap.read()
    count += 1
  file = os.path.basename(source)
  print('Finished ', file)

source ='./midia/016-2017.mp4'
#video_to_frame(source)
video_to_frame_cropped(source)

# Rascunho recortar mascara de borda
#  #_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
#  #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#  #cnt = contours[0]
#  #x,y,w,h = cv2.boundingRect(cnt)
#  #crop = image[y:y+h,x:x+w]
#  #crop = thresh
#  #cv2.imwrite("fps/frame%d.png" % count, crop)     # save frame as JPEG file
#  cv.imwrite("fps/frame%d.png" % count, gray)     # save frame as JPEG file
