import face_recognition
import numpy as np

def get_features(img):
	return face_recognition.face_encodings(img)

def get_face(features, embeddings):

	dist = 10e-5
	bindex = -1

	for i, val in enumerate(embeddings['features']):
		for f in val:
			ddist = face_recognition.face_distance(np.array(f), np.array(features))
			if ddist[0] <= 0.6 and ddist[0]< ddist:
				dist = ddist[0]
				bindex = i

		return embeddings['name'][bindex] if bindex>= 0 else None

if __name__ == "__main__":
	with open('features.json', 'r') as file:
		print(get_face(file['features'][-1][-1], file))

