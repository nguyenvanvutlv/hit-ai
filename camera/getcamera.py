from pathlib import Path
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
import numpy as np
from django.conf import settings
from camera.models import Image
from insightface.app import FaceAnalysis



    
class VideoCamera(object):
    def __init__(self):
        # self.person = cv2.imread("img/main.jpg")
        self.video = cv2.VideoCapture(0)
        self.app = FaceAnalysis(name = 'buffalo_sc')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        frame_flip = cv2.flip(image, 1)
        faces = self.app.get(frame_flip)
        
        for index, value in enumerate(faces):
            points = value['bbox'].astype(np.int)
            frame_flip = cv2.rectangle(frame_flip, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 4)
            cv2.putText(frame_flip, "A", (points[0], points[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        
        return jpeg.tobytes()
