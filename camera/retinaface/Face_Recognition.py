import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#---------------------------

import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

from .model import arcface, retinaface_model
from .commons import preprocess, postprocess, distance as dst

#---------------------------

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 2:
    import logging
    tf.get_logger().setLevel(logging.ERROR)

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image

#---------------------------

def build_model():
    
    global model #singleton design pattern
    
    if not "model" in globals():
        
        model = tf.function(
            retinaface_model.build_model(),
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

def preprocess_face(img, target_size=(224, 224), grayscale = False, model = None, enforce_detection = True, return_facial_area = False, align = True):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = get_image(img)
	base_img = img.copy()
	img, facial_area = detect_face(img, threshold=0.9, model = model, align = align, allow_upscaling = True)

	if type(img) != np.ndarray and img == None:
		return None, None

	#--------------------------

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape

	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)

		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			# Put the base image in the middle of the padded image
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------

	#double check: if target image is not still the same size with target.
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	#---------------------------------------------------

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	if return_facial_area == True:
		return img_pixels, facial_area
	else:
		return img_pixels

def normalize_input(img, normalization = 'base'):

	#issue 131 declares that some normalization techniques improves the accuracy

	if normalization == 'base':
		return img
	else:
		#@trevorgribble and @davedgd contributed this feature

		img *= 255 #restore input in scale of [0, 255] because it was normalized in scale of  [0, 1] in preprocess_face

		if normalization == 'raw':
			pass #return just restored pixels

		elif normalization == 'Facenet':
			mean, std = img.mean(), img.std()
			img = (img - mean) / std

		elif(normalization=="Facenet2018"):
			# simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
			img /= 127.5
			img -= 1

		elif normalization == 'VGGFace':
			# mean subtraction based on VGGFace1 training data
			img[..., 0] -= 93.5940
			img[..., 1] -= 104.7624
			img[..., 2] -= 129.1863

		elif(normalization == 'VGGFace2'):
			# mean subtraction based on VGGFace2 training data
			img[..., 0] -= 91.4953
			img[..., 1] -= 103.8827
			img[..., 2] -= 131.0912

		elif(normalization == 'ArcFace'):
			#Reference study: The faces are cropped and resized to 112×112,
			#and each pixel (ranged between [0, 255]) in RGB images is normalised
			#by subtracting 127.5 then divided by 128.
			img -= 127.5
			img /= 128

	#-----------------------------

	return img

def detect_face(img_path, threshold=0.9, model = None, align=True, allow_upscaling = True):
    img = get_image(img_path)

    obj = detect_faces(img, threshold=threshold, model = model, allow_upscaling = allow_upscaling)

    if type(obj) == dict and len(obj) > 0:
        identity = obj['face_1']
        facial_area = identity["facial_area"]

        #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
        detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

        if align == True:
            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            # mouth_right = landmarks["mouth_right"]
            # mouth_left = landmarks["mouth_left"]

            detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)

        return detected_face, facial_area
    else: #len(obj) == 0
        face = None
        facial_area = [0, 0, img.shape[0], img.shape[1]]

    return face, facial_area

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
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
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
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
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
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

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
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
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
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

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

                facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye, nose)

            resp.append(facial_img[:, :, ::-1])
    #elif type(obj) == tuple:

    return resp

def find_input_shape(model):

	#face recognition models have different size of inputs
	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	#----------------------
	#issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
	#whereas its older versions expect (y, x)

	if tf_major_version == 2 and tf_minor_version >= 5:
		x = input_shape[0]; y = input_shape[1]
		input_shape = (y, x)

	#----------------------

	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape

def represent(img_path, model = None, detector_backend = None, enforce_detection = True, align = True, normalization = 'base', return_facial_area = False):

	"""
	This function represents facial images as vectors.

	Parameters:
		img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.

		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

		normalization (string): normalize the input image before feeding to model

	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
	"""

	if model is None:
		model = arcface.loadModel()

	#---------------------------------

	#decide input shape
	input_shape_x, input_shape_y = find_input_shape(model)

	#detect and align
	img, facial_area = preprocess_face(img = img_path
		, model = detector_backend
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, align = align
		, return_facial_area=True)

	if type(img) != np.ndarray and img == None:
		return None, None

	#---------------------------------
	#custom normalization

	img = normalize_input(img = img, normalization = normalization)

	#---------------------------------

	#represent
	embedding = model.predict(img)[0].tolist()

	if return_facial_area == True:
		return embedding, facial_area
	else:
		return embedding

def initialize_input(img1_path, img2_path = None):

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False

		if (
			(type(img2_path) == str and img2_path != None) #exact image path, base64 image
			or (isinstance(img2_path, np.ndarray) and img2_path.any()) #numpy array
		):
			img_list = [[img1_path, img2_path]]
		else: #analyze function passes just img1_path
			img_list = [img1_path]

	return img_list, bulkProcess

def verify(img1_path, img2_path=None, distance_metric = 'cosine', model = None, detector_backend = None, enforce_detection = True, align = True, prog_bar = True, normalization = 'base'):

	"""
	This function verifies an image pair is same person or different persons.

	Parameters:
		img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

		e.g. img1_path = [
			['img1.jpg', 'img2.jpg'],
			['img2.jpg', 'img3.jpg']
		]

		distance_metric (string) or list: cosine, euclidean, euclidean_l2

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

		enforce_detection (boolean): If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.

		prog_bar (boolean): enable/disable a progress bar

	Returns:
		Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

		{
			"verified": True
			, "distance": 0.2563
			, "max_threshold_to_verify": 0.40
			, "similarity_metric": "cosine"
		}

	"""

	#------------------------------

	img_list, bulkProcess = initialize_input(img1_path, img2_path)

	resp_objects = []

	if type(distance_metric) == str:
		metrics = []
		metrics.append(distance_metric)
	else:
		metrics = distance_metric
		bulkProcess = True

	#------------------------------

	disable_option = (False if len(img_list) > 1 else True) or not prog_bar
	pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = disable_option)

	for index in pbar:

		instance = img_list[index]

		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]; img2_path = instance[1]

			img1_representation = represent(img_path = img1_path
					, model = model
					, detector_backend = detector_backend
					, enforce_detection = enforce_detection
					, align = align
					, normalization = normalization
					)

			img2_representation = represent(img_path = img2_path
					, model = model
					, enforce_detection = enforce_detection
					, detector_backend = detector_backend
					, align = align
					, normalization = normalization
					)

				#----------------------
				#find distances between embeddings
			
			if img1_representation == None or img2_representation == None:
				return None

			for j in metrics:
				if j == 'cosine':
					distance = dst.findCosineDistance(img1_representation, img2_representation)
				elif j == 'euclidean':
					distance = dst.findEuclideanDistance(img1_representation, img2_representation)
				elif j == 'euclidean_l2':
					distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
				else:
					raise ValueError("Invalid distance_metric passed - ", distance_metric)

				distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
				#----------------------
				#decision

				threshold = dst.findThreshold(j)

				if distance <= threshold:
					identified = True
				else:
					identified = False

				resp_obj = {
					"verified": identified
					, "distance": distance
					, "threshold": threshold
					, "similarity_metric": distance_metric
				}

				if bulkProcess == True:
					resp_objects.append(resp_obj)
				else:
					return resp_obj

		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)

	#-------------------------

	if bulkProcess == True:

		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["pair_%d" % (i+1)] = resp_item

		return resp_obj

def verify_database(img_path, distance_metric = 'cosine', model = None, detector_backend = None, enforce_detection = True, align = True, prog_bar = True, normalization = 'base'):

	"""
	This function verifies an image pair is same person or different persons.

	Parameters:
		img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed. 

		distance_metric (string) or list: cosine, euclidean, euclidean_l2

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

		enforce_detection (boolean): If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.

		prog_bar (boolean): enable/disable a progress bar

	Returns:
		Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

		{
			"verified": True
			, "distance": 0.2563
			, "max_threshold_to_verify": 0.40
			, "similarity_metric": "cosine"
		}

	"""

	img1_representation, region = represent(img_path = img_path
			, model = model
			, detector_backend = detector_backend
			, enforce_detection = enforce_detection
			, align = align
			, normalization = normalization
			, return_facial_area=True)

	if img1_representation == None:
		return None

	resp_objects = []
	with open('database.txt', 'r') as f:
		while True:
			line = f.readline()

			if not line:
				break
			
			line = line.split()
			label = line[0]

			img2_representation = list(map(float, line[1:]))

			#----------------------
			#find distances between embeddings
			if distance_metric == 'cosine':
				distance = dst.findCosineDistance(img1_representation, img2_representation)
			elif distance_metric == 'euclidean':
				distance = dst.findEuclideanDistance(img1_representation, img2_representation)
			elif distance_metric == 'euclidean_l2':
				distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
			else:
				raise ValueError("Invalid distance_metric passed - ", distance_metric)

			distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
			#----------------------
			#decision

			threshold = dst.findThreshold(distance_metric)

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