from imutils import paths
import imutils
import pickle
import cv2
import os
import face_recognition

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images('/directory/of/training/sets/of/images'))
knownEncodings=[]
knownNames=[]
#loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    #extrating the person name from directory address
	name = imagePath.split(os.path.sep)[-2]
	print(imagePath)
    #reading each image using opencv library
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Face detection using face_recognition library
	boxes = face_recognition.face_locations(rgb,model="hog")
    #encoding detected face into 128d vector
	encodings = face_recognition.face_encodings(rgb, boxes)
    #store the encoded vector and name of the person
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] serializing encodings...")
#creating dictionary where each encoding correspond to a name.
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
