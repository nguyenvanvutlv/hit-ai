import cv2
import matplotlib.pyplot as plt
import numpy as np
from camera.retinaface import Face_Recognition
from insightface.app import FaceAnalysis

from deepface.commons import functions
from deepface.basemodels import ArcFace
from deepface import DeepFace
model_arc = ArcFace.loadModel()


def getImG(img, points):
    points = points.astype(np.int)
    img = img[points[1]: points[3], points[0]: points[2]]
    return img

app = FaceAnalysis("buffalo_sc")
app.prepare(ctx_id= 0)




img1 = cv2.imread("img/source/nhat.png")
img2 = cv2.imread("img/source/hieu.png")
img3 = cv2.imread("img/source/khanh.png")
img4 = cv2.imread("img/source/vu.png")
# print(img4)



embed1 = getImG(img1, app.get(img1)[0]['bbox'])
embed2 = getImG(img2, app.get(img2)[0]['bbox'])
embed3 = getImG(img3, app.get(img3)[0]['bbox'])
embed4 = getImG(img4, app.get(img4)[0]['bbox'])


embed1 = DeepFace.represent(img_path = embed1, model_name = 'ArcFace', model = model_arc, enforce_detection = False)
embed2 = DeepFace.represent(img_path = embed2, model_name = 'ArcFace', model = model_arc, enforce_detection = False)
embed3 = DeepFace.represent(img_path = embed3, model_name = 'ArcFace', model = model_arc, enforce_detection = False)
embed4 = DeepFace.represent(img_path = embed4, model_name = 'ArcFace', model = model_arc, enforce_detection = False)

# print(embed1)


with open("img/source/database.txt", "a") as f:
    f.write("longnhat ")
    for value in embed1:
        f.write(str(value) + " ")
    f.write('\n')
    f.write("hieu ")
    for value in embed2:
        f.write(str(value) + " ")
    f.write('\n')
    f.write("khanh ")
    for value in embed3:
        f.write(str(value) + " ")
    f.write('\n')
    f.write("vu ")
    for value in embed3:
        f.write(str(value) + " ")
    f.write('\n')
