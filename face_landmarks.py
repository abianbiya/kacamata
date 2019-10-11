#!/usr/bin/env python
# coding: utf-8

# ## Facial Landmarks

# - install opencv 2
# conda install -c conda-forge opencv

# - Install Dlib
# pip install dlib

# - Install imutils
# pip install imutils

import time

import cv2
import dlib
import imutils
from imutils import face_utils
#from imutils.video import VideoStream
from videocaptureasync import VideoCaptureAsync
import numpy as np

print("[INFO] loading facial landmark predictor...")
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
#vs = VideoStream(src=0).start()
vs = VideoCaptureAsync(0)
vs.start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    
#    frame = vs.read()
    _, frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

        # loop over the face detections
        for rect in rects:
            # compute the bounding box of the face and draw it on the
            # frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                          (0, 255, 0), 1)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # hitung jarak-jarak
            lebarWajah = np.linalg.norm(shape[0] - shape[16])
            jarak2 = np.linalg.norm(shape[3] - shape[13])
            jarak3 = np.linalg.norm(shape[5] - shape[11])
            tinggiWajah = np.linalg.norm((shape[27][0], shape[17][1]) - shape[8])
            indeksMorfo = tinggiWajah / lebarWajah * 100
            
            cv2.putText(frame, "jarak-jarak: {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}".format(lebarWajah, jarak2, jarak3, tinggiWajah), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(frame, "Indeks Morfologi: {:0.2f}".format(indeksMorfo), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            for (i, (x, y)) in enumerate(shape):
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()