from pathlib import Path
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
from django.conf import settings
from .retinaface import Face_Recognition
from camera.models import Image


def getImageFromDatabase(img):
    resp = Face_Recognition.verify_database(img)
    if resp == None:
    	return None, None, None
    labels = resp['label']
    area = resp['facial_area']
    distance = resp['distance']
    if distance > 0.5:
    	labels = "Unknown"
    return labels, area, distance

    
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
        labels, area, distance = getImageFromDatabase(frame_flip)
        print(labels, distance)
        if labels != None:
        	cv2.rectangle(frame_flip, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), 4)
        	cv2.putText(frame_flip, labels, (area[0], area[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        
        return jpeg.tobytes()
