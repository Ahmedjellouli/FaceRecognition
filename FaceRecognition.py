import cv2
from Recognizer import *

# for more details on how to use this code see : https://github.com/Ahmedjellouli/FaceDetection


# Image = Image(Recognizer = Recognizer(Database="empty", Tolerance=0.55, detectFrontalFace=False, detectLandmarks=True),
#               filename = "Faces\elon.jpg" ,
#               Save = True )

Video = Video(Recognizer = Recognizer(Database="Database", Tolerance=0.55, detectFrontalFace=False, detectLandmarks=True),
              filename = "Videos\elon.mp4" ,  # put your image path here e.g : D:\image.jpg
               )
Video.RecognizeFaces()   #to detect faces in image
Video.AddAudio()
# Image.RecognizeFaces()
