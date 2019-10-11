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
import imutils
#import os
from imutils import face_utils
from videocaptureasync import VideoCaptureAsync
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


# target_image = '../images/obama.jpg'
# target_image = '195502221980032002.jpg'
target_image = 'F56.jpg'
#target_image = '../DataFotoWajah/F2.jpg'
image = cv2.imread(target_image)

landmarks, rects = get_landmarks(image)
(bX, bY, bW, bH) = face_utils.rect_to_bb(rects[0])

lebarWajah = np.linalg.norm(landmarks[0] - landmarks[16])
jarak2 = np.linalg.norm(landmarks[2] - landmarks[14])
jarak3 = np.linalg.norm(landmarks[5] - landmarks[11])
tinggiWajah = np.linalg.norm(np.matrix((landmarks[27,0], landmarks[17,1])) - landmarks[8])
indeksMorfo = tinggiWajah / lebarWajah * 100

print("lebar: {:0.2f}, tinggi: {:0.2f}, indeks: {:0.2f}".format(lebarWajah, tinggiWajah, indeksMorfo))
print(landmarks.shape)
print(str(landmarks[17]))
print(str(landmarks[17]).split(' '))


cv2.putText(image, "jarak-jarak: {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}".format(lebarWajah, tinggiWajah, jarak2, jarak3), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.putText(image, "Lebar : {:0.2f}".format(lebarWajah), (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.putText(image, "Panjang: {:0.2f}".format(tinggiWajah), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.putText(image, "Jarak_2: {:0.2f}".format(jarak2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.putText(image, "Jarak_3: {:0.2f}".format(jarak3), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
cv2.putText(image, "Indeks Morfologi: {:0.2f}".format(indeksMorfo), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

image_with_landmarks = annotate_landmarks(image, landmarks)

cv2.imshow('Result', image_with_landmarks)
cv2.imwrite('image_with_landmarks_F21.jpg', image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()
