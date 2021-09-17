import time
import cv2
import dlib
import numpy as np
from moviepy.editor import *
import screeninfo


class Image:
    def __init__(self, Recognizer, filename, Save=False):
        self.Image =  filename
        self.Recognizer = Recognizer
        self.Save = Save


    def RecognizeFaces(self):
        self.Image = cv2.imread(self.Image, cv2.IMREAD_UNCHANGED)
        self.__resize()
        self.Recognizer.Process(self.Image)

        cv2.imshow('Faces in this image', self.Recognizer.frame)

        if self.Save:
            cv2.imwrite('Faces in this image.png', self.Image)
        cv2.waitKey()

    def __resize(self):
        screen = screeninfo.get_monitors()[0]
        Imgwidth = self.Image.shape[1]
        screenwidth, screenheight = screen.width - 100, screen.height - 100
        if Imgwidth > screenwidth:
            fx = screenwidth / Imgwidth
            self.Image = cv2.resize(self.Image, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        Imgheight = self.Image.shape[0]
        if Imgheight > screenheight:
            fx = screenheight / Imgheight
            self.Image = cv2.resize(self.Image, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)


class Video:
    def __init__(self, filename, Recognizer):
        self.Video =  filename
        self.VideoDst =  filename.replace(".mp4" ,"[Face detected].mp4"  )
        self.Recognizer = Recognizer

    def RecognizeFaces(self):
        videoCapture = cv2.VideoCapture(self.Video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        videoWriter = cv2.VideoWriter(self.VideoDst, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        success, self.frame = videoCapture.read()
        length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        FrameCount = 0
        oldProcessValue = 0
        bars = ""
        InitTime = time.time()
        print(success)
        while success:  # Loop until there are no more frames.
            self.Recognizer.Process(self.frame)
            videoWriter.write(self.frame)
            success, self.frame = videoCapture.read()
            FrameCount += 1

            process = 100 + ((FrameCount - length) / length) * 100

            if int(process) != oldProcessValue:
                oldProcessValue = int(process)
                Fps = float(FrameCount / (time.time() - InitTime))
                estimateTime = (length - FrameCount) / Fps
                h = int(estimateTime / (60 * 60))
                min = int((estimateTime - 60 * 60 * h) / 60)
                s = (estimateTime - min * 60 - h * 60 * 60)

                Fps = "{:.2f}".format(Fps)
                bars = bars + "|"
                sys.stdout.write('\r ' + "%02d" % (oldProcessValue,) + "%")
                print(bars + " " * (100 - oldProcessValue) + "| [" + str(Fps) + "frame/s  " + "%02d" % (
                h,) + ":" + "%02d" % (min,) + ":" + "%02d" % (s,) + " ]", sep='', end='', flush=True)
        videoCapture.release()
        # self.AddAudio()

    def AddAudio(self):
        # extract Audio from original video
        video = VideoFileClip(os.path.join("", "", self.Video))
        video.audio.write_audiofile(os.path.join("", "", "sound.mp3"))

        videoclip = VideoFileClip(self.VideoDst)
        audioclip = AudioFileClip( "sound.mp3")

        videoclip.audio = CompositeAudioClip([audioclip])
        videoFile = str(self.VideoDst).lower()
        videoFile = videoFile.replace(".mp4", "1.mp4")
        videoclip.write_videofile(videoFile)


class Recognizer:

    def __init__(self, Database="Database", Tolerance=0.55, detectFrontalFace=False, detectLandmarks=False):
        self.Database = Database
        self.detectFrontalFace = detectFrontalFace
        self.Tolerance = Tolerance
        self.detectLandmarks = detectLandmarks

        self.predictorModel = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
        self.faceRecoModel = dlib.face_recognition_model_v1("Models/dlib_face_recognition_resnet_model_v1.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.frame = None
        self.DatabaseFacesList = []
        self.DatabaseNamesList = []
        self.DataBaseFacesTo_128P()


    def DataBaseFacesTo_128P(self):

        for _, _, filenames in os.walk(self.Database):
            for filename in filenames:
                base = os.path.basename(filename)
                Name = os.path.splitext(base)[0]
                self.DatabaseNamesList.append(Name)
                sys.stdout.write('\r[INFO] coding database faces...' + (filename))
                img = cv2.imread(self.Database + "\\"+filename)
                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                FaceRect = self.detector(grayImg)
                shape = self.predictorModel(image=grayImg, box=FaceRect[0])
                self.DatabaseFacesList.append( np.array(self.faceRecoModel.compute_face_descriptor(img, shape, num_jitters=1)))

        print("\n ")
        self.DatabaseFacesList = np.array(self.DatabaseFacesList)
    def AddName(self, description, org):
        font = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        Y = 20
        x = 165
        y =-40
        Skip = -20
        intro = ""
        intro_ = ""
        description = description+"-"
        for letter in description:   #to organize writing name and description
                    intro = intro + letter
                    if letter == '-':
                        Skip += 20
                        intro = ""
                    if intro == '':
                         cv2.putText(self.frame, intro_, (org[0] - x, org[1] + 2 * Y - y+Skip), font, fontScale, color, thickness, cv2.LINE_AA)
                    intro_ = intro
        cv2.line(self.frame,(org[0]-x, org[1]+ 2*Y -y+15+Skip),(org[0]-x +300, org[1]+ 2*Y -y+15+Skip), (150, 150, 28), 2 )

    def AddLandmarks(self, landmarks):
        if self.detectLandmarks:
            for Point in range(1, 68):
                x = landmarks.part(Point).x
                y = landmarks.part(Point).y
                cv2.circle(self.frame, (x, y), 0, (179, 135,47 ), 2)

        if self.detectFrontalFace:
            x1  = landmarks.part(1).x
            y20 = landmarks.part(20).y
            x17 = landmarks.part(15).x
            y9  = landmarks.part(8).y
            cv2.rectangle(self.frame, (x1, y20 -100), (x17, y9+25), color=(150, 150, 28), thickness=1)

    def Process(self, frame) :
        self.frame = frame
        grayImg = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        FaceRects = self.detector(grayImg)

        for FaceRect in FaceRects:

            shape = self.predictorModel(image=grayImg, box=FaceRect)

            imgFace128 =  self.faceRecoModel.compute_face_descriptor(self.frame, shape, num_jitters=1)
            norms = np.linalg.norm(self.DatabaseFacesList - imgFace128, axis=1)
            i = 0
            ChangedTolerance = self.Tolerance
            FaceName = ""
            for norm in norms:
                if norm < ChangedTolerance:  # to take the minimum value
                    ChangedTolerance = norm
                    FaceName = self.DatabaseNamesList[i]
                i += 1
            self.AddName( description=FaceName, org=(shape.part(8).x, shape.part(8).y))
            self.AddLandmarks(shape)
