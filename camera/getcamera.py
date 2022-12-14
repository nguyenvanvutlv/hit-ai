from pathlib import Path
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
import numpy as np
from django.conf import settings
from camera.models import Image
from insightface.app import FaceAnalysis
from .retinaface.model import arcface
from .retinaface.commons import distance as dst
from deepface import DeepFace
from deepface.basemodels import ArcFace


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
    

def verify_database(embedding1, region, distance_metric = 'cosine'):
    resp_objects = []
    with open('img/source/database.txt', 'r') as f:
        while True:
            line = f.readline()
            
            if not line:
                break
            line = line.split()
            label = line[0]
            embedding2 = list(map(float, line[1:]))
            if distance_metric == 'cosine':
                distance = dst.findCosineDistance(embedding1, embedding2)
            elif distance_metric == 'euclidean':
                distance = dst.findEuclideanDistance(embedding1, embedding2)
            elif distance_metric == 'euclidean_l2':
                distance = dst.findEuclideanDistance(dst.l2_normalize(embedding1), dst.l2_normalize(embedding2))
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)
            
            distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
            print(distance)
            threshold = dst.findThreshold(distance_metric)
            # print(threshold)
            
            if distance <= threshold:
                resp_obj = {
					"label": label
					, "facial_area": region
					, "distance": distance
					, "threshold": threshold
					, "similarity_metric": distance_metric
				}
                resp_objects.append(resp_obj)
    if len(resp_objects) == 0:
        return None
    
    return min(resp_objects, key=lambda x: x['distance'])


class VideoCamera(object):
    def __init__(self):
        # self.person = cv2.imread("img/main.jpg")
        self.video = cv2.VideoCapture(0)
        frame_width = int(self.video.get(3))
        frame_height = int(self.video.get(4))  
        self.app = FaceAnalysis(name = 'buffalo_sc')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.model_arc = ArcFace.loadModel()
        
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        frame_flip = cv2.flip(image, 1)
        faces = self.app.get(frame_flip)
        for index, value in enumerate(faces):
            points = value['bbox'].astype(np.int)
            result = None
            # try:
            #     embedding = DeepFace.represent(img_path = frame_flip, model_name = 'ArcFace', model = self.model_arc, enforce_detection = False)
            #     
            # except:
            #     print()
            copy_frame = frame_flip.copy()
            copy_frame = copy_frame[points[1]: points[3], points[0]: points[2]]
            result = DeepFace.represent(img_path = copy_frame, model_name = 'ArcFace', model = self.model_arc, enforce_detection = False)
            result = verify_database(result, points)

            
            label = "Unknown" if result == None else result['label']
                       
            draw_border(frame_flip, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 3, 10, 20)
            
            cv2.putText(frame_flip, label, (points[0], points[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        
        return jpeg.tobytes()
