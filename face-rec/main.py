import argparse
import cv2
import numpy as np
import json
from find_faces import findFace
from find_faces import get_bbox
from landmarks import align_face
from feature_extraction import get_features
from feature_extraction import get_face

ap = argparse.ArgumentParser()
ap.add_argument("file", help="path to the image")
ap.add_argument("-n", "--new", help="name of the new face")
ap.add_argument("-u","--update",help="name of the face to be updates")
args = vars(ap.parse_args())

with open('features.json','r') as file:
	embeddings = json.load(file)

img = cv2.imread(args['file'])
face = findFace(img)
alignFace = align_face(img, face)
features = get_features(alignFace)

if args['new'] and features:
	embeddings['name'].append(args['new'])
	embeddings['features'].append([list(features[0])])

	with open('features.json', 'w') as file:
		json.dump(embeddings, file)

elif args['update'] and features:
	index = embeddings['name'].index(args['update'])
	embeddings['features'][index].append(list(features[0]))

	with open('features.json', 'w') as file:
		json.dump(embeddings, file)

elif features:
	name = get_face(features, embeddings)
	out = get_bbox(img, face, name)
	cv2.imshow('output', out)
	cv2.waitKey(0)
	cv2.destoryAllWindows()


