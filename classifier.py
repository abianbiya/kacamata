import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import cv2
import dlib
import numpy as np
import imutils
# import os
from imutils import face_utils
from videocaptureasync import VideoCaptureAsync
pd.set_option('display.max_columns', None)


class classifier:

    classify = None
    jk = None

    def __init__(self, jk, km):
        clf = DecisionTreeClassifier(max_depth=6, random_state=1234)
        if(km == '1'):
            input_file = "data_mentah_baca.csv"
        else:
            input_file = "data_mentah_sung.csv"

        # %% baca data
        df = pd.read_csv(input_file, sep=',')
        le_kelamin = LabelEncoder()
        le_bingkai = LabelEncoder()

        # convert categorical column to numeric
        # df['kelamin'] = le_kelamin.fit_transform(df['kelamin'])
        # df['bingkai'] = le_bingkai.fit_transform(df['bingkai'])

        X = df.values[:, 2:-1]
        Y = df.values[:, -1]

        # normalisasi
        # scaler_training = StandardScaler()
        # X[:, 0:-1] = scaler_training.fit_transform(X[:, 0:-1])

        print(df)
        # exit()
        # pecah jadi data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1234)
        clf.fit(X_train, y_train.astype('int'))
        
        
        self.classify = clf
        self.jk = jk


    def predict(self, path):
        image = cv2.imread(path)
        # cv2.imshow("Imagew",image)
        # cv2.waitKey(0)

        iz, landmarks, rects = self.get_landmarks(image)
        if iz==False:
            return rects, 0, 0

        (bX, bY, bW, bH) = face_utils.rect_to_bb(rects[0])

        lebarWajah = np.linalg.norm(landmarks[0] - landmarks[16])
        jarak2 = np.linalg.norm(landmarks[2] - landmarks[14])
        jarak3 = np.linalg.norm(landmarks[5] - landmarks[11])
        tinggiWajah = np.linalg.norm(np.matrix((landmarks[27, 0], landmarks[17, 1])) - landmarks[8])
        indeksMorfo = tinggiWajah / lebarWajah * 100

        print('Lebar Wajah: '+str(lebarWajah))
        print('Jarak2: '+str(jarak2))
        print('Jarak3: '+str(jarak3))
        print('Tinggi Wajah: '+str(tinggiWajah))
        print('Index Morfologi: '+str(indeksMorfo))

        hasil = self.classify.predict(np.array([[lebarWajah, tinggiWajah, jarak2, jarak3, indeksMorfo, self.jk]]))
        # print('hasil klasifikasi: '+str(hasil)

        return hasil[0], landmarks[17], landmarks[26]


    def get_landmarks(self, im):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()
        rects = detector(im, 1)

        if len(rects) > 1:
            return False, 'Too many faces', 99
        if len(rects) == 0:
            return False, 'No Face', 98

        return True, np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]), rects

    def annotate_landmarks(self, im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,

                        color=(0, 0, 255))
            cv2.circle(im, pos, 1, color=(0, 255, 255))
        return im

# cl = classifier()
# print(cl.predict('uploads/F68.jpg'))
