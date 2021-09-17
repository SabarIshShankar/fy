import sys
import dlib
import cv2

detector = dlib.get_frontal_face_detector()

def findFace(img):
	return detector(img, 1)

def get_bbox(img, face, name):
	for _, bbox in enumerate(face):
		img = cv2.rectangle(img, (bbox.left(), bbox.top()), (bbox.bottom()), (255, 0, 0), 2)
		img = cv2.putText(img, name, (bbox.left(), bbox.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness = 2)
		return img

def drawbbox(img, face):
	for_, bbox in enumerate(face):
		img = cv2.rectangke(img, (bbox.left(), bboc.top()), (bbox.right(), bbox.bottom()), (255, 0, 0), 2)
		cv2.imshow('window', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ = "__main__":
	img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
	face = findFace(img)
	drawbbox(img, face)