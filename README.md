# Recognize Faces in video or image easily 

in this code we will take any video or image and try to Recognize all faces on them using **OpenCV** and **dlib** libraries;

![](FacialReco.png)

# Prerequisites

after all make sure that you have already install the folowing libraries:

- opencv
```
pip install opencv-python
```
- dlib "Require Cmake"
```
pip install cmake
```
```
pip install dlib
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
## **Database** Folder
To determine the faces that our program should know
- Put the **face image** as **png** or **jpg** format in **Database** Folder
- using the name of the image:
  you can add The name and also a small description as follow : 
  ```
  Name1-Name2-Name3-First description line- second description line- third description line- ... -.jpg
  ```
  **"-"** used to write in new line
  Example: 
  ```
  Angela-Merkel-[German politician serving- as Chancellor of Germany- since 2005]-.jpg
  ```
## code 

- Fonctionnalities avaliable in this code:

interesting instance :
```
Recognizer = Recognizer(Database="Database",
                        Tolerance=0.55,
                        detectFrontalFace=False, 
                        detectLandmarks=True)

Image = Image(Recognizer=Recognizer,
              filename="Faces\\angela-merkel.jpeg",
              Save=True)

Video = Video(Recognizer=Recognizer,
              filename="Videos\elon.mp4",   
              )
```

``` 
        - detectFrontalFace     = True or false to detect frontale face in video or Image
        - Tolerance             = 0 to 1 to determine tolerance of Recognition        
        - detectLandmarks       = True or false to detect 68 landmarks in video or Image 
        - save                  = True or false to save image after Recognition
        
``` 
 
- start detection with :
``` 
        - Image.RecognizeFaces() : to Recognitize faces in image
        - Video.RecognizeFaces() : to Recognitize faces in Video
        - Video.AddAudio()       : to add audio to Video after face Recognition
``` 
# Author

* **Ahmed Jellouli** - *ENGINEERING STUDENT* 

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


