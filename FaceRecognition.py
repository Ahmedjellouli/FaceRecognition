import cv2
from Recognizer import *

# for more details on how to use this code see : https://github.com/Ahmedjellouli/FaceRecognition

Recognizer = Recognizer(Database="Database",
                        Tolerance=0.55,
                        detectFrontalFace=False,
                        detectLandmarks=True)

Image = Image(Recognizer=Recognizer,
              filename="Faces\\Malala-Yousafzai.jpg",
              Save=True)

Video = Video(Recognizer=Recognizer,
              filename="Videos\elon.mp4",  # put your image path here e.g : D:\image.jpg
              )

Image.RecognizeFaces()
Video.RecognizeFaces()  # to detect faces in image
Video.AddAudio()
