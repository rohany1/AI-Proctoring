# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

class eye_tracker:
    def __init__(self):
    # frames the eye must be below the threshold
        self.EYE_AR_THRESH = 0.35
        self.EYE_AR_CONSEC_FRAMES = 3

        # initialize the frame counters and the total number of blinks


        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_68.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    def get_eye(self,frame):
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        ori=frame.copy()
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]


            # average the eye aspect ratio together for both eyes


            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if np.array_equal(ori,frame):
            return frame,False
        return frame,True

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame

import cv2

if __name__ == "__main__":
    eye = eye_tracker()
    cap = cv2.VideoCapture(0)
    counter=0

    while 1:
        ret, frame = cap.read()
        frame,state = eye.get_eye(frame)
        cv2.imshow('frame', frame)
        if not state:
            counter+=1
            if counter==30:
                print("No eyes-found")
                counter=0
        else:
            counter=0
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # if the `q` key was pressed, break from the loop


