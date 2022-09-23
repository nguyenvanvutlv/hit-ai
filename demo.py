import cv2, json, os, time
import numpy as np
from camera.hawk_eyes.face import RetinaFace, ArcFace, Landmark
from camera.hawk_eyes.tracking import BYTETracker


"""
        load model
"""
# ---------------------------------------------- #
retina = RetinaFace(model_name='retina_s')
bt = BYTETracker()
arc = ArcFace(model_name='arcface_s')
landmark = Landmark()
# ---------------------------------------------- #


# ---------------------------------------------- #
"""
        load database
"""
database_emb = {
    'userID': [],
    'embs': []
}
img_data_list = os.listdir("img/source")
for i in range(len(img_data_list)):
    img_path = os.path.join("img/source", img_data_list[i])
    img = cv2.imread(img_path)
    # print(img_path)
    label = img_data_list[i].split('.')[0]
    fbox, kpss = retina.detect(img)
    emb = arc.get(img, kpss[0])
    database_emb['embs'].append(emb)
    database_emb['userID'].append(label)
print("Load data done!")
# ---------------------------------------------- #




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

current_frame = 0
last_frame = 0



video = cv2.VideoCapture("img/A.mp4")
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
_fps = video.get(cv2.CAP_PROP_FPS)
result = cv2.VideoWriter('img/out.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         _fps, size)


while True:
    current_frame = time.time()
    _, frame = video.read()
    if _ != True:
        break
    # frame = cv2.flip(frame, 1)
    bboxes, kpss = retina.detect(frame)
    """
        bboxes[i]: danh sách box có mặt 
        kpss[i]  : danh sách phân tích mặt
    """
    boxes, tids = bt.predict(frame, bboxes)
    """
        boxes[i]: dạnh sách box có mặt di chuyển đến vị trí tiếp theo từ bboxes[i]
    """
    
    tkpss = [None]*len(bboxes)
    for i in range(len(boxes)):
        min_d = 9e5
        tb = boxes[i]
        for j in range(len(bboxes)):
            fb = bboxes[j]
            d = abs(tb[0]-fb[0])+abs(tb[1]-fb[1]) + abs(tb[2]-fb[2])+abs(tb[3]-fb[3])
            if d < min_d:
                min_d = d
                tkpss[i] = kpss[j]
    
    embs = []
    ids = []
    for tid, tbox, tkps in zip(tids, boxes, tkpss):
            # print(localtime)
        box = tbox[:4].astype(int)
        land = landmark.get(frame, tbox)
        angle = landmark.get_face_angle(frame, land, False)
        if abs(angle) < 15:
            embs.append(arc.get(frame, tkps))
            ids.append(tid)
        draw_border(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 10, 20)
        
    
    if len(embs) > 0:
        for i in embs:
            dis_cosin = np.dot(database_emb['embs'], i) / (np.linalg.norm(database_emb['embs']) * np.linalg.norm(i))
            label = "Stranger" if max(dis_cosin) <= 0.1 else str(database_emb['userID'][np.argmax(dis_cosin)])
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    fps = 1/(current_frame - last_frame)
    last_frame = current_frame
    current_frame = time.time()
    
    cv2.putText(frame, 'fps: {}'.format(int(fps)),  (frame.shape[1]-150, frame.shape[0]-100), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2 )
    
    # cv2.imshow('image', frame)
    result.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
video.release()
result.release()
cv2.destroyAllWindows()