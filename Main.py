import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import dlib
from scipy.spatial import distance
import time

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

paths = 'Drowsiness_Detection'
mylist = os.listdir(paths)  # Paths
images = []  # Contains Images
className = []  # Contains Names
for cl in mylist:
    curImg = cv2.imread(f'{paths}/{cl}')   # Curr Image
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])  # Name of Curr Image
print(className)
print(mylist)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def Student_name(name):
    with open('Students.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            nows = datetime.now()
            dtString = nows.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

ans = False

while True:
    success, img = cap.read()
    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = hog_face_detector(imgS)

    for face in faces:
        face_landmarks = dlib_facelandmark(imgS, face)
        leftEye = []
        rightEye = []
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        if EAR < 0.26:
            # nows = datetime.now()
            # start_time = nows.strftime("%S")
            # start_time = int(start_time)
            start_time = time.time()

            while True:
                # now = datetime.now()
                # end_time = now.strftime("%S")
                # end_time = int(end_time)
                end_time = time.time()
                if EAR >= 0.26:
                    break

                if end_time - start_time > 8:
                    ans = True
                    break


    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if ans == True:
                Student_name(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
