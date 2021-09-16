# Detect Faces in video or image easily 

in this code we will take any video or image and try to detect all faces on them using **OpenCV** and **dlib** libraries;

![](FacialReco.png)

# Prerequisites

after all make sure that you have already install the folowing libraries:

- opencv
```
pip install opencv-python
```
- dlib "Require Cmake"
```
pip install dlib
```
```
pip install cmake
```
- time
```
pip install time
```
- moviepy
```
pip install moviepy
```
- screeninfo
```
pip install screeninfo
```
# How to use this code

- put in "path" video/image the directory where your Video/image exist:

```
VideoPath ="D:\XX\XX\FaceDetection\Videos\\" #video folder path
ImagePath ="D:\xx\xx\FaceDetection\Faces\\" #Image folder path

``` 

- Fonctionnalities avaliable in this code:

interesting instance :
```
Video = detectInVideo(detectors = detectors(frame=None , detectFFace=False , detectEyes=False ,
                                             detectLandmarks=True),
                      Path = VideoPath ,
                      Video = "... .mp4"  # put your Video name Video.mp4)

Image = Image(detectors = detectors(frame=None ,  detectFFace=True , detectEyes=True ,detectLandmarks=True ),
              Path=ImagePath ,
              Image = "Dev Patel.png" ,  # put your image path here e.g : D:\image.jpg
              Save = True
              )
```

``` 
        - detectFFace     = True or false to detect frontale face in video or Image
        - detectEyes      = True or false to detect eyes in video or Image
        - detectLandmarks = True or false to detect 68 landmarks in video or Image 
        - save            = True or false to save image after detecion
        
``` 
 
- start detection with :
``` 
        - Image.detectFaces() : to detect faces in image
        - Video.detectFaces() : to detect faces in Video
        - Video.AddAudio()     : to add audio to Video after face detection
``` 
# Author

* **Ahmed Jellouli** - *ENGINEERING STUDENT* 

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


