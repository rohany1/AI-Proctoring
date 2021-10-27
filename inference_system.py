from person_detection import person_and_phone_detection
from eye_tracker import eye_tracker
import cv2
class inference:
    def __init__(self):
        self.PersonAndPhone = person_and_phone_detection()
        self.EyeGaze = eye_tracker()
        self.eye_counter=0

        self.cap = cv2.VideoCapture(0)


    def infer(self):
        PersonAndPhone=self.PersonAndPhone
        EyeGaze=self.EyeGaze
        cap=self.cap
        while True:
            ret, frame = cap.read()
            if ret == False:
                break

            eye_gaze_bool=EyeGaze.infer(frame)
            flag = PersonAndPhone.infer(frame)

            if 0 in flag:
                print('Mobile Phone detected')
            if 1 in flag:
                print('No person detected')
            if 2 in flag:
                print('More than one person detected')
            if not eye_gaze_bool:
                self.eye_counter+=1
                if self.eye_counter==5:
                    self.eye_counter=0
                    print("No eyes Detected")
            if eye_gaze_bool and self.eye_counter>0:
                self.eye_counter=0

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
inf=inference()
inf.infer()
