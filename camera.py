import cv2
#face_cascade=cv2.CascadeClassifier()
class VideoCamera(object):

    #intitalizing the camera on the browser
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    
    #free the camera from the browser when its not required
    def __del__(self):
        self.video.releast()

    #read the frame and send it to browser
    def get_frame(self):
        ret, frame=self.video.read()
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #convert the frame given in matrices to jpg form
        ret, jpeg=cv2.imencode('.jpg',frame)
        #converting jpeg to byte format given to browser
        return jpeg.tobytes()