from pathlib import Path
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
from django.conf import settings
from .retinaface import Face_Recognition
from camera.models import Image


def getImageFromDatabase():
    Humans = Image.objects.all()
    Humans = [{'labels' : str(i.title), 'src' : str(i.file)} for i in Humans]
    
    
class VideoCamera(object):
    def __init__(self):
        # self.person = cv2.imread("img/main.jpg")
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        frame_flip = cv2.flip(image, 1)
        frame_flip = cv2.resize(frame_flip, (640, 480))
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        
        return jpeg.tobytes()
