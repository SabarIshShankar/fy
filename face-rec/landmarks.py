import dlib
import cv2
import openface

predictor_model = "shape_predictor_68_face_landmarks.dat"

pose_predictor = dlib.shape_predictor(predictor_model)
aligner = openface.AlignDlib(predictor_model)

def align_face(img, rect):
	for _, bbox in enumerate(rect):
		landmarks = pose_predictor(img, bbox)
		alignedFace = aligner.align(500, img, bbox, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		return alignedFace