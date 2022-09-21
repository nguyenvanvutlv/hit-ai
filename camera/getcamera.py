from pathlib import Path
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
import numpy as np
from django.conf import settings
from camera.models import Image
from insightface.app import FaceAnalysis
from .retinaface.model import arcface 
from .retinaface import Face_Recognition

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    
    
class VideoCamera(object):
    def __init__(self):
        # self.person = cv2.imread("img/main.jpg")
        self.video = cv2.VideoCapture(0)
        self.app = FaceAnalysis(name = 'buffalo_sc')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.arcface_model = arcface.loadModel()
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        frame_flip = cv2.flip(image, 1)
        faces = self.app.get(frame_flip)
        for index, value in enumerate(faces):
            points = value['bbox'].astype(np.int)
            
            
            crop_frame = frame_flip.copy()
            crop_frame = crop_frame[points[1]: points[3], points[0]: points[2]]
            # cv2.imwrite("img/test.jpg", crop_frame)
            
            # vectorEmbedding = Face_Recognition.represent(crop_frame, model = self.arcface_model)
            # print(vectorEmbedding)
            
            
            # frame_flip = cv2.rectangle(frame_flip, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 4)
            draw_border(frame_flip, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 3, 10, 20)
            cv2.putText(frame_flip, "A", (points[0], points[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        
        return jpeg.tobytes()
