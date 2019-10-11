#!/usr/bin/env python
# coding: utf-8

# ## Facial Landmarks

# Using Dlib
# Dlib is a novel object detector library based on this journal: https://arxiv.org/abs/1502.00046
# (Max-Margin Object Detector)

# - Install Dlib
# pip install dlib

import cv2
import dlib
import numpy as np
import os
from imutils import face_utils
from imutils import paths
import csv

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]), rects


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,

                    color=(0, 0, 255))
        cv2.circle(im, pos, 1, color=(0, 255, 255))
    return im


target_dir = 'tambahan/'
# target_image = '../images/obama.jpg'
# target_image = '195502221980032002.jpg'
#target_image = 'DataFotoWajah/F9.jpg'
#target_image = 'DataFotoWajah/F2.jpg'

image_paths = sorted(list(paths.list_images(target_dir)))
idxs = list(range(0, len(image_paths)))

csv_file = open("auto_calculation_face.csv", mode="w", newline='')
fieldnames = ['filename', 'lebar', 'panjang', 'jarak2', 'jarak3', 'indeks']
csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
csv_writer.writeheader()

for idx in idxs:
    image_path = image_paths[idx]
    filename = image_path.split('/')[-1]
    
    image = cv2.imread(image_path)
    landmarks, rects = get_landmarks(image)
    (bX, bY, bW, bH) = face_utils.rect_to_bb(rects[0])
    
    lebarWajah = np.linalg.norm(landmarks[0] - landmarks[16])
    jarak2 = np.linalg.norm(landmarks[2] - landmarks[14])
    jarak3 = np.linalg.norm(landmarks[5] - landmarks[11])
    tinggiWajah = np.linalg.norm(np.matrix((landmarks[27,0], landmarks[17,1])) - landmarks[8])
    indeksMorfo = tinggiWajah / lebarWajah * 100
    
    print("file:{}, lebar: {:0.2f}, tinggi: {:0.2f}, indeks: {:0.2f}".format(filename, lebarWajah, tinggiWajah, indeksMorfo))
    csv_writer.writerow({'filename': filename, 'lebar': lebarWajah, 'panjang': tinggiWajah, 'jarak2': jarak2, 'jarak3': jarak3, 'indeks': indeksMorfo})
    
csv_file.close()
    

#image_with_landmarks = annotate_landmarks(image, landmarks)
#
#cv2.imshow('Result', image_with_landmarks)
#cv2.imwrite('image_with_landmarks.jpg', image_with_landmarks)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
