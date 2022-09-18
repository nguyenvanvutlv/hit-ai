# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import cv2
import gdown
from pathlib import Path
from PIL import Image
import math
from imutils.video import VideoStream
import imutils
import cv2, os, urllib.request
from django.conf import settings

#---------------------------

tf_version = int(tf.__version__.split(".")[0])

if tf_version == 2:
    import logging
    tf.get_logger().setLevel(logging.ERROR)

#---------------------------


if tf_version == 1:
    from keras.models import Model
    from keras.layers import Input, BatchNormalization, ZeroPadding2D, Conv2D, ReLU, MaxPool2D, Add, UpSampling2D, concatenate, Softmax

else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, BatchNormalization, ZeroPadding2D, Conv2D, ReLU, MaxPool2D, Add, UpSampling2D, concatenate, Softmax

# ----------------------------
# commons 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(img, left_eye, right_eye, nose):

    #this function aligns given face in img based on left and right eye coordinates

    #left eye is the eye appearing on the left (right eye of the person)
    #left top point is (0, 0)

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    #decide the image is inverse

    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))
    
    if False:

        img = cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 2, (0, 255, 255), 2)
        img = cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 2, (255, 0, 0), 2)
        img = cv2.circle(img, center_eyes, 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(nose[0]), int(nose[1])), 2, (255, 255, 255), 2)

    #-----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------

    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        
        #PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        #In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))
        
        
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

        if center_eyes[1] > nose[1]:
            img = Image.fromarray(img)
            img = np.array(img.rotate(180))

    #-----------------------

    return img #return img anyway

#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def bbox_pred(boxes, box_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1]>4:
        pred_boxes[:,4:] = box_deltas[:,4:]

    return pred_boxes

# This function copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
      return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
    return pred

# This function copied from rcnn module of retinaface-tf2 project: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/processing/bbox_transform.py
def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

#this function is mainly based on the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/anchors.pyx
def anchors_plane(height, width, stride, base_anchors):
    A = base_anchors.shape[0]
    c_0_2 = np.tile(np.arange(0, width)[np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
    c_1_3 = np.tile(np.arange(0, height)[:, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
    all_anchors = np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(base_anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1))
    return all_anchors

#this function is mainly based on the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/rcnn/cython/cpu_nms.pyx
#Fast R-CNN by Ross Girshick
def cpu_nms(dets, threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]; iy1 = y1[i]; ix2 = x2[i]; iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j]); yy1 = max(iy1, y1[j]); xx2 = min(ix2, x2[j]); yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1); h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep

# ----------------------------
# commons 2
def resize_image(img, scales, allow_upscaling):
    img_h, img_w = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )

    return img, im_scale


# This function is modified from the following code snippet:
# https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L76-L96
def preprocess_image(img, allow_upscaling):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [1024, 1980]

    img, im_scale = resize_image(img, scales, allow_upscaling)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]

    return im_tensor, img.shape[0:2], im_scale

# ----------------------------

def load_weights(model):

    home = str(os.getenv('DEEPFACE_HOME', default=Path.home()))

    exact_file = home+'/.deepface/weights/retinaface.h5'
    #url = 'https://drive.google.com/file/d/1K3Eq2k1b9dpKkucZjPAiCCnNzfCMosK4'
    #url = 'https://drive.google.com/uc?id=1K3Eq2k1b9dpKkucZjPAiCCnNzfCMosK4'
    url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5'

    #-----------------------------

    if not os.path.exists(home+"/.deepface"):
        os.mkdir(home+"/.deepface")
        print("Directory ",home,"/.deepface created")

    if not os.path.exists(home+"/.deepface/weights"):
        os.mkdir(home+"/.deepface/weights")
        print("Directory ",home,"/.deepface/weights created")

    #-----------------------------

    if os.path.isfile(exact_file) != True:
        print("retinaface.h5 will be downloaded from the url "+url)
        gdown.download(url, exact_file, quiet=False)

    #-----------------------------

    #gdown should download the pretrained weights here. If it does not still exist, then throw an exception.
    if os.path.isfile(exact_file) != True:
        raise ValueError("Pre-trained weight could not be loaded!"
            +" You might try to download the pre-trained weights from the url "+ url
            + " and copy it to the ", exact_file, "manually.")

    model.load_weights(exact_file)

    return model

def build_model():

    data = Input(dtype=tf.float32, shape=(None, None, 3), name='data')

    bn_data = BatchNormalization(epsilon=1.9999999494757503e-05, name='bn_data', trainable=False)(data)

    conv0_pad = ZeroPadding2D(padding=tuple([3, 3]))(bn_data)

    conv0 = Conv2D(filters = 64, kernel_size = (7, 7), name = 'conv0', strides = [2, 2], padding = 'VALID', use_bias = False)(conv0_pad)

    bn0 = BatchNormalization(epsilon=1.9999999494757503e-05, name='bn0', trainable=False)(conv0)

    relu0 = ReLU(name='relu0')(bn0)

    pooling0_pad = ZeroPadding2D(padding=tuple([1, 1]))(relu0)

    pooling0 = MaxPool2D((3, 3), (2, 2), padding='VALID', name='pooling0')(pooling0_pad)

    stage1_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit1_bn1', trainable=False)(pooling0)

    stage1_unit1_relu1 = ReLU(name='stage1_unit1_relu1')(stage1_unit1_bn1)

    stage1_unit1_conv1 = Conv2D(filters = 64, kernel_size = (1, 1), name = 'stage1_unit1_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit1_relu1)

    stage1_unit1_sc = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage1_unit1_sc', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit1_relu1)

    stage1_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit1_bn2', trainable=False)(stage1_unit1_conv1)

    stage1_unit1_relu2 = ReLU(name='stage1_unit1_relu2')(stage1_unit1_bn2)

    stage1_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit1_relu2)

    stage1_unit1_conv2 = Conv2D(filters = 64, kernel_size = (3, 3), name = 'stage1_unit1_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit1_conv2_pad)

    stage1_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit1_bn3', trainable=False)(stage1_unit1_conv2)

    stage1_unit1_relu3 = ReLU(name='stage1_unit1_relu3')(stage1_unit1_bn3)

    stage1_unit1_conv3 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage1_unit1_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit1_relu3)

    plus0_v1 = Add()([stage1_unit1_conv3 , stage1_unit1_sc])

    stage1_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit2_bn1', trainable=False)(plus0_v1)

    stage1_unit2_relu1 = ReLU(name='stage1_unit2_relu1')(stage1_unit2_bn1)

    stage1_unit2_conv1 = Conv2D(filters = 64, kernel_size = (1, 1), name = 'stage1_unit2_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit2_relu1)

    stage1_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit2_bn2', trainable=False)(stage1_unit2_conv1)

    stage1_unit2_relu2 = ReLU(name='stage1_unit2_relu2')(stage1_unit2_bn2)

    stage1_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit2_relu2)

    stage1_unit2_conv2 = Conv2D(filters = 64, kernel_size = (3, 3), name = 'stage1_unit2_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit2_conv2_pad)

    stage1_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit2_bn3', trainable=False)(stage1_unit2_conv2)

    stage1_unit2_relu3 = ReLU(name='stage1_unit2_relu3')(stage1_unit2_bn3)

    stage1_unit2_conv3 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage1_unit2_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit2_relu3)

    plus1_v2 = Add()([stage1_unit2_conv3 , plus0_v1])

    stage1_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit3_bn1', trainable=False)(plus1_v2)

    stage1_unit3_relu1 = ReLU(name='stage1_unit3_relu1')(stage1_unit3_bn1)

    stage1_unit3_conv1 = Conv2D(filters = 64, kernel_size = (1, 1), name = 'stage1_unit3_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit3_relu1)

    stage1_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit3_bn2', trainable=False)(stage1_unit3_conv1)

    stage1_unit3_relu2 = ReLU(name='stage1_unit3_relu2')(stage1_unit3_bn2)

    stage1_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit3_relu2)

    stage1_unit3_conv2 = Conv2D(filters = 64, kernel_size = (3, 3), name = 'stage1_unit3_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit3_conv2_pad)

    stage1_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage1_unit3_bn3', trainable=False)(stage1_unit3_conv2)

    stage1_unit3_relu3 = ReLU(name='stage1_unit3_relu3')(stage1_unit3_bn3)

    stage1_unit3_conv3 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage1_unit3_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage1_unit3_relu3)

    plus2 = Add()([stage1_unit3_conv3 , plus1_v2])

    stage2_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit1_bn1', trainable=False)(plus2)

    stage2_unit1_relu1 = ReLU(name='stage2_unit1_relu1')(stage2_unit1_bn1)

    stage2_unit1_conv1 = Conv2D(filters = 128, kernel_size = (1, 1), name = 'stage2_unit1_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit1_relu1)

    stage2_unit1_sc = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage2_unit1_sc', strides = [2, 2], padding = 'VALID', use_bias = False)(stage2_unit1_relu1)

    stage2_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit1_bn2', trainable=False)(stage2_unit1_conv1)

    stage2_unit1_relu2 = ReLU(name='stage2_unit1_relu2')(stage2_unit1_bn2)

    stage2_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit1_relu2)

    stage2_unit1_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'stage2_unit1_conv2', strides = [2, 2], padding = 'VALID', use_bias = False)(stage2_unit1_conv2_pad)

    stage2_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit1_bn3', trainable=False)(stage2_unit1_conv2)

    stage2_unit1_relu3 = ReLU(name='stage2_unit1_relu3')(stage2_unit1_bn3)

    stage2_unit1_conv3 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage2_unit1_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit1_relu3)

    plus3 = Add()([stage2_unit1_conv3 , stage2_unit1_sc])

    stage2_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit2_bn1', trainable=False)(plus3)

    stage2_unit2_relu1 = ReLU(name='stage2_unit2_relu1')(stage2_unit2_bn1)

    stage2_unit2_conv1 = Conv2D(filters = 128, kernel_size = (1, 1), name = 'stage2_unit2_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit2_relu1)

    stage2_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit2_bn2', trainable=False)(stage2_unit2_conv1)

    stage2_unit2_relu2 = ReLU(name='stage2_unit2_relu2')(stage2_unit2_bn2)

    stage2_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit2_relu2)

    stage2_unit2_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'stage2_unit2_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit2_conv2_pad)

    stage2_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit2_bn3', trainable=False)(stage2_unit2_conv2)

    stage2_unit2_relu3 = ReLU(name='stage2_unit2_relu3')(stage2_unit2_bn3)

    stage2_unit2_conv3 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage2_unit2_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit2_relu3)

    plus4 = Add()([stage2_unit2_conv3 , plus3])

    stage2_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit3_bn1', trainable=False)(plus4)

    stage2_unit3_relu1 = ReLU(name='stage2_unit3_relu1')(stage2_unit3_bn1)

    stage2_unit3_conv1 = Conv2D(filters = 128, kernel_size = (1, 1), name = 'stage2_unit3_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit3_relu1)

    stage2_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit3_bn2', trainable=False)(stage2_unit3_conv1)

    stage2_unit3_relu2 = ReLU(name='stage2_unit3_relu2')(stage2_unit3_bn2)

    stage2_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit3_relu2)

    stage2_unit3_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'stage2_unit3_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit3_conv2_pad)

    stage2_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit3_bn3', trainable=False)(stage2_unit3_conv2)

    stage2_unit3_relu3 = ReLU(name='stage2_unit3_relu3')(stage2_unit3_bn3)

    stage2_unit3_conv3 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage2_unit3_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit3_relu3)

    plus5 = Add()([stage2_unit3_conv3 , plus4])

    stage2_unit4_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit4_bn1', trainable=False)(plus5)

    stage2_unit4_relu1 = ReLU(name='stage2_unit4_relu1')(stage2_unit4_bn1)

    stage2_unit4_conv1 = Conv2D(filters = 128, kernel_size = (1, 1), name = 'stage2_unit4_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit4_relu1)

    stage2_unit4_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit4_bn2', trainable=False)(stage2_unit4_conv1)

    stage2_unit4_relu2 = ReLU(name='stage2_unit4_relu2')(stage2_unit4_bn2)

    stage2_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit4_relu2)

    stage2_unit4_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'stage2_unit4_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit4_conv2_pad)

    stage2_unit4_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage2_unit4_bn3', trainable=False)(stage2_unit4_conv2)

    stage2_unit4_relu3 = ReLU(name='stage2_unit4_relu3')(stage2_unit4_bn3)

    stage2_unit4_conv3 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage2_unit4_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage2_unit4_relu3)

    plus6 = Add()([stage2_unit4_conv3 , plus5])

    stage3_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit1_bn1', trainable=False)(plus6)

    stage3_unit1_relu1 = ReLU(name='stage3_unit1_relu1')(stage3_unit1_bn1)

    stage3_unit1_conv1 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage3_unit1_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit1_relu1)

    stage3_unit1_sc = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit1_sc', strides = [2, 2], padding = 'VALID', use_bias = False)(stage3_unit1_relu1)

    stage3_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit1_bn2', trainable=False)(stage3_unit1_conv1)

    stage3_unit1_relu2 = ReLU(name='stage3_unit1_relu2')(stage3_unit1_bn2)

    stage3_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit1_relu2)

    stage3_unit1_conv2 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'stage3_unit1_conv2', strides = [2, 2], padding = 'VALID', use_bias = False)(stage3_unit1_conv2_pad)

    ssh_m1_red_conv = Conv2D(filters = 256, kernel_size = (1, 1), name = 'ssh_m1_red_conv', strides = [1, 1], padding = 'VALID', use_bias = True)(stage3_unit1_relu2)

    stage3_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit1_bn3', trainable=False)(stage3_unit1_conv2)

    ssh_m1_red_conv_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m1_red_conv_bn', trainable=False)(ssh_m1_red_conv)

    stage3_unit1_relu3 = ReLU(name='stage3_unit1_relu3')(stage3_unit1_bn3)

    ssh_m1_red_conv_relu = ReLU(name='ssh_m1_red_conv_relu')(ssh_m1_red_conv_bn)

    stage3_unit1_conv3 = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit1_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit1_relu3)

    plus7 = Add()([stage3_unit1_conv3 , stage3_unit1_sc])

    stage3_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit2_bn1', trainable=False)(plus7)

    stage3_unit2_relu1 = ReLU(name='stage3_unit2_relu1')(stage3_unit2_bn1)

    stage3_unit2_conv1 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage3_unit2_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit2_relu1)

    stage3_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit2_bn2', trainable=False)(stage3_unit2_conv1)

    stage3_unit2_relu2 = ReLU(name='stage3_unit2_relu2')(stage3_unit2_bn2)

    stage3_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit2_relu2)

    stage3_unit2_conv2 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'stage3_unit2_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit2_conv2_pad)

    stage3_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit2_bn3', trainable=False)(stage3_unit2_conv2)

    stage3_unit2_relu3 = ReLU(name='stage3_unit2_relu3')(stage3_unit2_bn3)

    stage3_unit2_conv3 = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit2_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit2_relu3)

    plus8 = Add()([stage3_unit2_conv3 , plus7])

    stage3_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit3_bn1', trainable=False)(plus8)

    stage3_unit3_relu1 = ReLU(name='stage3_unit3_relu1')(stage3_unit3_bn1)

    stage3_unit3_conv1 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage3_unit3_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit3_relu1)

    stage3_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit3_bn2', trainable=False)(stage3_unit3_conv1)

    stage3_unit3_relu2 = ReLU(name='stage3_unit3_relu2')(stage3_unit3_bn2)

    stage3_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit3_relu2)

    stage3_unit3_conv2 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'stage3_unit3_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit3_conv2_pad)

    stage3_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit3_bn3', trainable=False)(stage3_unit3_conv2)

    stage3_unit3_relu3 = ReLU(name='stage3_unit3_relu3')(stage3_unit3_bn3)

    stage3_unit3_conv3 = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit3_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit3_relu3)

    plus9 = Add()([stage3_unit3_conv3 , plus8])

    stage3_unit4_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit4_bn1', trainable=False)(plus9)

    stage3_unit4_relu1 = ReLU(name='stage3_unit4_relu1')(stage3_unit4_bn1)

    stage3_unit4_conv1 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage3_unit4_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit4_relu1)

    stage3_unit4_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit4_bn2', trainable=False)(stage3_unit4_conv1)

    stage3_unit4_relu2 = ReLU(name='stage3_unit4_relu2')(stage3_unit4_bn2)

    stage3_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit4_relu2)

    stage3_unit4_conv2 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'stage3_unit4_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit4_conv2_pad)

    stage3_unit4_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit4_bn3', trainable=False)(stage3_unit4_conv2)

    stage3_unit4_relu3 = ReLU(name='stage3_unit4_relu3')(stage3_unit4_bn3)

    stage3_unit4_conv3 = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit4_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit4_relu3)

    plus10 = Add()([stage3_unit4_conv3 , plus9])

    stage3_unit5_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit5_bn1', trainable=False)(plus10)

    stage3_unit5_relu1 = ReLU(name='stage3_unit5_relu1')(stage3_unit5_bn1)

    stage3_unit5_conv1 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage3_unit5_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit5_relu1)

    stage3_unit5_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit5_bn2', trainable=False)(stage3_unit5_conv1)

    stage3_unit5_relu2 = ReLU(name='stage3_unit5_relu2')(stage3_unit5_bn2)

    stage3_unit5_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit5_relu2)

    stage3_unit5_conv2 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'stage3_unit5_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit5_conv2_pad)

    stage3_unit5_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit5_bn3', trainable=False)(stage3_unit5_conv2)

    stage3_unit5_relu3 = ReLU(name='stage3_unit5_relu3')(stage3_unit5_bn3)

    stage3_unit5_conv3 = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit5_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit5_relu3)

    plus11 = Add()([stage3_unit5_conv3 , plus10])

    stage3_unit6_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit6_bn1', trainable=False)(plus11)

    stage3_unit6_relu1 = ReLU(name='stage3_unit6_relu1')(stage3_unit6_bn1)

    stage3_unit6_conv1 = Conv2D(filters = 256, kernel_size = (1, 1), name = 'stage3_unit6_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit6_relu1)

    stage3_unit6_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit6_bn2', trainable=False)(stage3_unit6_conv1)

    stage3_unit6_relu2 = ReLU(name='stage3_unit6_relu2')(stage3_unit6_bn2)

    stage3_unit6_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit6_relu2)

    stage3_unit6_conv2 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'stage3_unit6_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit6_conv2_pad)

    stage3_unit6_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage3_unit6_bn3', trainable=False)(stage3_unit6_conv2)

    stage3_unit6_relu3 = ReLU(name='stage3_unit6_relu3')(stage3_unit6_bn3)

    stage3_unit6_conv3 = Conv2D(filters = 1024, kernel_size = (1, 1), name = 'stage3_unit6_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage3_unit6_relu3)

    plus12 = Add()([stage3_unit6_conv3 , plus11])

    stage4_unit1_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit1_bn1', trainable=False)(plus12)

    stage4_unit1_relu1 = ReLU(name='stage4_unit1_relu1')(stage4_unit1_bn1)

    stage4_unit1_conv1 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage4_unit1_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit1_relu1)

    stage4_unit1_sc = Conv2D(filters = 2048, kernel_size = (1, 1), name = 'stage4_unit1_sc', strides = [2, 2], padding = 'VALID', use_bias = False)(stage4_unit1_relu1)

    stage4_unit1_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit1_bn2', trainable=False)(stage4_unit1_conv1)

    stage4_unit1_relu2 = ReLU(name='stage4_unit1_relu2')(stage4_unit1_bn2)

    stage4_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit1_relu2)

    stage4_unit1_conv2 = Conv2D(filters = 512, kernel_size = (3, 3), name = 'stage4_unit1_conv2', strides = [2, 2], padding = 'VALID', use_bias = False)(stage4_unit1_conv2_pad)

    ssh_c2_lateral = Conv2D(filters = 256, kernel_size = (1, 1), name = 'ssh_c2_lateral', strides = [1, 1], padding = 'VALID', use_bias = True)(stage4_unit1_relu2)

    stage4_unit1_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit1_bn3', trainable=False)(stage4_unit1_conv2)

    ssh_c2_lateral_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_c2_lateral_bn', trainable=False)(ssh_c2_lateral)

    stage4_unit1_relu3 = ReLU(name='stage4_unit1_relu3')(stage4_unit1_bn3)

    ssh_c2_lateral_relu = ReLU(name='ssh_c2_lateral_relu')(ssh_c2_lateral_bn)

    stage4_unit1_conv3 = Conv2D(filters = 2048, kernel_size = (1, 1), name = 'stage4_unit1_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit1_relu3)

    plus13 = Add()([stage4_unit1_conv3 , stage4_unit1_sc])

    stage4_unit2_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit2_bn1', trainable=False)(plus13)

    stage4_unit2_relu1 = ReLU(name='stage4_unit2_relu1')(stage4_unit2_bn1)

    stage4_unit2_conv1 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage4_unit2_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit2_relu1)

    stage4_unit2_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit2_bn2', trainable=False)(stage4_unit2_conv1)

    stage4_unit2_relu2 = ReLU(name='stage4_unit2_relu2')(stage4_unit2_bn2)

    stage4_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit2_relu2)

    stage4_unit2_conv2 = Conv2D(filters = 512, kernel_size = (3, 3), name = 'stage4_unit2_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit2_conv2_pad)

    stage4_unit2_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit2_bn3', trainable=False)(stage4_unit2_conv2)

    stage4_unit2_relu3 = ReLU(name='stage4_unit2_relu3')(stage4_unit2_bn3)

    stage4_unit2_conv3 = Conv2D(filters = 2048, kernel_size = (1, 1), name = 'stage4_unit2_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit2_relu3)

    plus14 = Add()([stage4_unit2_conv3 , plus13])

    stage4_unit3_bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit3_bn1', trainable=False)(plus14)

    stage4_unit3_relu1 = ReLU(name='stage4_unit3_relu1')(stage4_unit3_bn1)

    stage4_unit3_conv1 = Conv2D(filters = 512, kernel_size = (1, 1), name = 'stage4_unit3_conv1', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit3_relu1)

    stage4_unit3_bn2 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit3_bn2', trainable=False)(stage4_unit3_conv1)

    stage4_unit3_relu2 = ReLU(name='stage4_unit3_relu2')(stage4_unit3_bn2)

    stage4_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit3_relu2)

    stage4_unit3_conv2 = Conv2D(filters = 512, kernel_size = (3, 3), name = 'stage4_unit3_conv2', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit3_conv2_pad)

    stage4_unit3_bn3 = BatchNormalization(epsilon=1.9999999494757503e-05, name='stage4_unit3_bn3', trainable=False)(stage4_unit3_conv2)

    stage4_unit3_relu3 = ReLU(name='stage4_unit3_relu3')(stage4_unit3_bn3)

    stage4_unit3_conv3 = Conv2D(filters = 2048, kernel_size = (1, 1), name = 'stage4_unit3_conv3', strides = [1, 1], padding = 'VALID', use_bias = False)(stage4_unit3_relu3)

    plus15 = Add()([stage4_unit3_conv3 , plus14])

    bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name='bn1', trainable=False)(plus15)

    relu1 = ReLU(name='relu1')(bn1)

    ssh_c3_lateral = Conv2D(filters = 256, kernel_size = (1, 1), name = 'ssh_c3_lateral', strides = [1, 1], padding = 'VALID', use_bias = True)(relu1)

    ssh_c3_lateral_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_c3_lateral_bn', trainable=False)(ssh_c3_lateral)

    ssh_c3_lateral_relu = ReLU(name='ssh_c3_lateral_relu')(ssh_c3_lateral_bn)

    ssh_m3_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_conv1 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'ssh_m3_det_conv1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_conv1_pad)

    ssh_m3_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_context_conv1 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m3_det_context_conv1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_context_conv1_pad)

    ssh_c3_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_c3_up")(ssh_c3_lateral_relu)

    ssh_m3_det_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m3_det_conv1_bn', trainable=False)(ssh_m3_det_conv1)

    ssh_m3_det_context_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv1_bn', trainable=False)(ssh_m3_det_context_conv1)

    x1_shape = tf.shape(ssh_c3_up)
    x2_shape = tf.shape(ssh_c2_lateral_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop0 = tf.slice(ssh_c3_up, offsets, size, "crop0")

    ssh_m3_det_context_conv1_relu = ReLU(name='ssh_m3_det_context_conv1_relu')(ssh_m3_det_context_conv1_bn)

    plus0_v2 = Add()([ssh_c2_lateral_relu , crop0])

    ssh_m3_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m3_det_context_conv1_relu)

    ssh_m3_det_context_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m3_det_context_conv2', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_context_conv2_pad)

    ssh_m3_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m3_det_context_conv1_relu)

    ssh_m3_det_context_conv3_1 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m3_det_context_conv3_1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_context_conv3_1_pad)

    ssh_c2_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus0_v2)

    ssh_c2_aggr = Conv2D(filters = 256, kernel_size = (3, 3), name = 'ssh_c2_aggr', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_c2_aggr_pad)

    ssh_m3_det_context_conv2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv2_bn', trainable=False)(ssh_m3_det_context_conv2)

    ssh_m3_det_context_conv3_1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv3_1_bn', trainable=False)(ssh_m3_det_context_conv3_1)

    ssh_c2_aggr_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_c2_aggr_bn', trainable=False)(ssh_c2_aggr)

    ssh_m3_det_context_conv3_1_relu = ReLU(name='ssh_m3_det_context_conv3_1_relu')(ssh_m3_det_context_conv3_1_bn)

    ssh_c2_aggr_relu = ReLU(name='ssh_c2_aggr_relu')(ssh_c2_aggr_bn)

    ssh_m3_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m3_det_context_conv3_1_relu)

    ssh_m3_det_context_conv3_2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m3_det_context_conv3_2', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_context_conv3_2_pad)

    ssh_m2_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_conv1 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'ssh_m2_det_conv1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_conv1_pad)

    ssh_m2_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_context_conv1 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m2_det_context_conv1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_context_conv1_pad)

    ssh_m2_red_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_m2_red_up")(ssh_c2_aggr_relu)

    ssh_m3_det_context_conv3_2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m3_det_context_conv3_2_bn', trainable=False)(ssh_m3_det_context_conv3_2)

    ssh_m2_det_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m2_det_conv1_bn', trainable=False)(ssh_m2_det_conv1)

    ssh_m2_det_context_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv1_bn', trainable=False)(ssh_m2_det_context_conv1)

    x1_shape = tf.shape(ssh_m2_red_up)
    x2_shape = tf.shape(ssh_m1_red_conv_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop1 = tf.slice(ssh_m2_red_up, offsets, size, "crop1")

    ssh_m3_det_concat = concatenate([ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn], 3, name='ssh_m3_det_concat')

    ssh_m2_det_context_conv1_relu = ReLU(name='ssh_m2_det_context_conv1_relu')(ssh_m2_det_context_conv1_bn)

    plus1_v1 = Add()([ssh_m1_red_conv_relu , crop1])

    ssh_m3_det_concat_relu = ReLU(name='ssh_m3_det_concat_relu')(ssh_m3_det_concat)

    ssh_m2_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m2_det_context_conv1_relu)

    ssh_m2_det_context_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m2_det_context_conv2', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_context_conv2_pad)

    ssh_m2_det_context_conv3_1_pad  = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m2_det_context_conv1_relu)

    ssh_m2_det_context_conv3_1 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m2_det_context_conv3_1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_context_conv3_1_pad)

    ssh_c1_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus1_v1)

    ssh_c1_aggr = Conv2D(filters = 256, kernel_size = (3, 3), name = 'ssh_c1_aggr', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_c1_aggr_pad)

    face_rpn_cls_score_stride32 = Conv2D(filters = 4, kernel_size = (1, 1), name = 'face_rpn_cls_score_stride32', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_concat_relu)

    inter_1 = concatenate([face_rpn_cls_score_stride32[:, :, :, 0], face_rpn_cls_score_stride32[:, :, :, 1]], axis=1)
    inter_2 = concatenate([face_rpn_cls_score_stride32[:, :, :, 2], face_rpn_cls_score_stride32[:, :, :, 3]], axis=1)
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride32 = tf.transpose(final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride32")

    face_rpn_bbox_pred_stride32 = Conv2D(filters = 8, kernel_size = (1, 1), name = 'face_rpn_bbox_pred_stride32', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_concat_relu)

    face_rpn_landmark_pred_stride32 = Conv2D(filters = 20, kernel_size = (1, 1), name = 'face_rpn_landmark_pred_stride32', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m3_det_concat_relu)

    ssh_m2_det_context_conv2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv2_bn', trainable=False)(ssh_m2_det_context_conv2)

    ssh_m2_det_context_conv3_1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv3_1_bn', trainable=False)(ssh_m2_det_context_conv3_1)

    ssh_c1_aggr_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_c1_aggr_bn', trainable=False)(ssh_c1_aggr)

    ssh_m2_det_context_conv3_1_relu = ReLU(name='ssh_m2_det_context_conv3_1_relu')(ssh_m2_det_context_conv3_1_bn)

    ssh_c1_aggr_relu = ReLU(name='ssh_c1_aggr_relu')(ssh_c1_aggr_bn)

    face_rpn_cls_prob_stride32 = Softmax(name = 'face_rpn_cls_prob_stride32')(face_rpn_cls_score_reshape_stride32)

    input_shape = [tf.shape(face_rpn_cls_prob_stride32)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride32[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride32[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride32[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride32[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride32")

    ssh_m2_det_context_conv3_2_pad  = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m2_det_context_conv3_1_relu)

    ssh_m2_det_context_conv3_2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m2_det_context_conv3_2', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_context_conv3_2_pad)

    ssh_m1_det_conv1_pad            = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_conv1 = Conv2D(filters = 256, kernel_size = (3, 3), name = 'ssh_m1_det_conv1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_conv1_pad)

    ssh_m1_det_context_conv1_pad    = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_context_conv1 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m1_det_context_conv1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_context_conv1_pad)

    ssh_m2_det_context_conv3_2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m2_det_context_conv3_2_bn', trainable=False)(ssh_m2_det_context_conv3_2)

    ssh_m1_det_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m1_det_conv1_bn', trainable=False)(ssh_m1_det_conv1)

    ssh_m1_det_context_conv1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv1_bn', trainable=False)(ssh_m1_det_context_conv1)

    ssh_m2_det_concat               = concatenate([ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn], 3, name='ssh_m2_det_concat')

    ssh_m1_det_context_conv1_relu = ReLU(name='ssh_m1_det_context_conv1_relu')(ssh_m1_det_context_conv1_bn)

    ssh_m2_det_concat_relu = ReLU(name='ssh_m2_det_concat_relu')(ssh_m2_det_concat)

    ssh_m1_det_context_conv2_pad    = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m1_det_context_conv1_relu)

    ssh_m1_det_context_conv2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m1_det_context_conv2', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_context_conv2_pad)

    ssh_m1_det_context_conv3_1_pad  = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m1_det_context_conv1_relu)

    ssh_m1_det_context_conv3_1 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m1_det_context_conv3_1', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_context_conv3_1_pad)

    face_rpn_cls_score_stride16 = Conv2D(filters = 4, kernel_size = (1, 1), name = 'face_rpn_cls_score_stride16', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_concat_relu)

    inter_1 = concatenate([face_rpn_cls_score_stride16[:, :, :, 0], face_rpn_cls_score_stride16[:, :, :, 1]], axis=1)
    inter_2 = concatenate([face_rpn_cls_score_stride16[:, :, :, 2], face_rpn_cls_score_stride16[:, :, :, 3]], axis=1)
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride16 = tf.transpose(final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride16")

    face_rpn_bbox_pred_stride16 = Conv2D(filters = 8, kernel_size = (1, 1), name = 'face_rpn_bbox_pred_stride16', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_concat_relu)

    face_rpn_landmark_pred_stride16 = Conv2D(filters = 20, kernel_size = (1, 1), name = 'face_rpn_landmark_pred_stride16', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m2_det_concat_relu)

    ssh_m1_det_context_conv2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv2_bn', trainable=False)(ssh_m1_det_context_conv2)

    ssh_m1_det_context_conv3_1_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv3_1_bn', trainable=False)(ssh_m1_det_context_conv3_1)

    ssh_m1_det_context_conv3_1_relu = ReLU(name='ssh_m1_det_context_conv3_1_relu')(ssh_m1_det_context_conv3_1_bn)

    face_rpn_cls_prob_stride16      = Softmax(name = 'face_rpn_cls_prob_stride16')(face_rpn_cls_score_reshape_stride16)

    input_shape = [tf.shape(face_rpn_cls_prob_stride16)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride16[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride16[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride16[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride16[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride16")

    ssh_m1_det_context_conv3_2_pad  = ZeroPadding2D(padding=tuple([1, 1]))(ssh_m1_det_context_conv3_1_relu)

    ssh_m1_det_context_conv3_2 = Conv2D(filters = 128, kernel_size = (3, 3), name = 'ssh_m1_det_context_conv3_2', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_context_conv3_2_pad)

    ssh_m1_det_context_conv3_2_bn = BatchNormalization(epsilon=1.9999999494757503e-05, name='ssh_m1_det_context_conv3_2_bn', trainable=False)(ssh_m1_det_context_conv3_2)

    ssh_m1_det_concat               = concatenate([ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn], 3, name='ssh_m1_det_concat')

    ssh_m1_det_concat_relu = ReLU(name='ssh_m1_det_concat_relu')(ssh_m1_det_concat)
    face_rpn_cls_score_stride8 = Conv2D(filters = 4, kernel_size = (1, 1), name = 'face_rpn_cls_score_stride8', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_concat_relu)

    inter_1 = concatenate([face_rpn_cls_score_stride8[:, :, :, 0], face_rpn_cls_score_stride8[:, :, :, 1]], axis=1)
    inter_2 = concatenate([face_rpn_cls_score_stride8[:, :, :, 2], face_rpn_cls_score_stride8[:, :, :, 3]], axis=1)
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride8 = tf.transpose(final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride8")

    face_rpn_bbox_pred_stride8 = Conv2D(filters = 8, kernel_size = (1, 1), name = 'face_rpn_bbox_pred_stride8', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_concat_relu)

    face_rpn_landmark_pred_stride8 = Conv2D(filters = 20, kernel_size = (1, 1), name = 'face_rpn_landmark_pred_stride8', strides = [1, 1], padding = 'VALID', use_bias = True)(ssh_m1_det_concat_relu)

    face_rpn_cls_prob_stride8       = Softmax(name = 'face_rpn_cls_prob_stride8')(face_rpn_cls_score_reshape_stride8)

    input_shape = [tf.shape(face_rpn_cls_prob_stride8)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride8[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride8[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride8[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride8[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride8")

    model = Model(inputs=data,
                    outputs=[face_rpn_cls_prob_reshape_stride32,
                                                   face_rpn_bbox_pred_stride32,
                                                   face_rpn_landmark_pred_stride32,
                                                   face_rpn_cls_prob_reshape_stride16,
                                                   face_rpn_bbox_pred_stride16,
                                                   face_rpn_landmark_pred_stride16,
                                                   face_rpn_cls_prob_reshape_stride8,
                                                   face_rpn_bbox_pred_stride8,
                                                   face_rpn_landmark_pred_stride8
                                                   ])
    model = load_weights(model)

    return model



# ----------------------------
def build_model_main():
    
    global model #singleton design pattern
    
    if not "model" in globals():
        
        model = tf.function(
            build_model(),
            input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),)
        )

    return model

def get_image(img_path):
    if type(img_path) == str:  # Load from file path
        if not os.path.isfile(img_path):
            raise ValueError("Input image file path (", img_path, ") does not exist.")
        img = cv2.imread(img_path)

    elif isinstance(img_path, np.ndarray):  # Use given NumPy array
        img = img_path.copy()

    else:
        raise ValueError("Invalid image input. Only file paths or a NumPy array accepted.")

    # Validate image shape
    if len(img.shape) != 3 or np.prod(img.shape) == 0:
        raise ValueError("Input image needs to have 3 channels at must not be empty.")

    return img

def detect_faces(img_path, threshold=0.9, model = None, allow_upscaling = True):
    """
    TODO: add function doc here
    """

    img = get_image(img_path)

    #---------------------------

    if model is None:
        model = build_model()

    #---------------------------

    nms_threshold = 0.4; decay4=0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
        'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
        'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
    }

    _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

    #---------------------------

    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_info, im_scale = preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _idx, s in enumerate(_feat_stride_fpn):
        _key = 'stride%s'%s
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s'%s]
        anchors = anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_deltas = bbox_deltas
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
        proposals = bbox_pred(anchors, bbox_deltas)

        proposals = clip_boxes(proposals, im_info[:2])

        if s==4 and decay4<1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel>=threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3]//A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
        landmarks = landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    if proposals.shape[0]==0:
        landmarks = np.zeros( (0,5,2) )
        return np.zeros( (0,5) ), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

    #nms = cpu_nms_wrapper(nms_threshold)
    #keep = nms(pre_det)
    keep = cpu_nms(pre_det, nms_threshold)

    det = np.hstack( (pre_det, proposals[:,4:]) )
    det = det[keep, :]
    landmarks = landmarks[keep]

    resp = {}
    for idx, face in enumerate(det):

        label = 'face_'+str(idx+1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))

        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

    return resp

def extract_faces(img_path, threshold=0.9, model = None, align = True, allow_upscaling = True):

    resp = []

    #---------------------------

    img = get_image(img_path)

    #---------------------------

    obj = detect_faces(img_path = img, threshold = threshold, model = model, allow_upscaling = allow_upscaling)

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]

            facial_area = identity["facial_area"]
            facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            if align == True:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                mouth_right = landmarks["mouth_right"]
                mouth_left = landmarks["mouth_left"]

                facial_img = alignment_procedure(facial_img, right_eye, left_eye, nose)

            resp.append(facial_img[:, :, ::-1])
    #elif type(obj) == tuple:

    return 





class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		# for (x, y, w, h) in faces_detected:
		# 	cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		frame_flip = cv2.flip(image,1)

		# faces_detected = detect_faces(frame_flip)

		# labels = list(faces_detected)

		# for index, value in enumerate(labels):
		# 	points = faces_detected[value]['facial_area']
		# 	frame_flip = cv2.rectangle(frame_flip, (points[0], points[1]), (points[2], points[3]), (255, 0, 0), 2)

		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()