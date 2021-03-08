# import the necessary packages
#from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import pickle
import time# import the necessary packages
#from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import pickle
import time
import cv2
import os
import face_recognition
import argparse

# construct the argument parser and parse the arguments
# use to get important argument for writing on surface
ap = argparse.ArgumentParser()
ap.add_argument("-vect1","--vect1", nargs="+", default=[0.9995409011029099, -0.004484674424180896, 0.029964557692245267])
ap.add_argument("-vect2","--vect2", nargs="+", default= [0.004409285053232823, 0.9999869468064428, 0.002581554192261182])
ap.add_argument("-vect3","--vect3", nargs="+", default=[-0.02997574398913453, -0.002448246727219587, 0.9995476281099668])
ap.add_argument("-orientation","--orientation", nargs="+", default=[90.01776885986328, -0.0022241799936940274, 90.00107320149739])
ap.add_argument("-start_position","--start_position", nargs="+", default= [0.47455549240112305, 0.14839012920856476, 0.06899100542068481])
ap.add_argument("-normal_vector","--normal_vector", nargs="+", default=[-0.0038400804315996684, -0.0003136357300199677, 0.12804830760992125])
ap.add_argument("-distance", type=str, default=0.0069057885984395145)
args = ap.parse_args()

# gstreamer plugins for linux for getting video frame from network
# resize the resolution for smoothness
gst="rtspsrc   location=rtsp://10.79.128.245/color   latency=5   !rtph264depay ! avdec_h264 ! videoscale!video/x-raw,width=480,height=270 ! videoconvert ! appsink "


# load the known faces and embeddings 
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("/home/dospina/Kinova_CV/encodings.pickle", "rb").read())


# Create the haar cascade for face detection from video stream
detector = cv2.CascadeClassifier('/home/dospina/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml')

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs =  cv2.VideoCapture(gst)

time.sleep(2.0)
# start the FPS counter
fps = FPS().start()
# loop over frames from the video file stream
while True:
    #capturing video using gstreamer plugins
	ret,frame = vs.read()

	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	# loop over the facial embeddings for face recognition 
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],encoding,0.55)
		name = "Unknown"
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

		# update the list of names
		names.append(name)
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
    # if the person is recognised the program that will write name will be called in the same terminal
	if len(names)!=0 and names[-1] != "Unknown":
		#command for running the program for writing name on board
		cmd="python3 /home/dospina/kortex/api_python/examples/102-Movement_high_level/Face_Recognition_Name_Final.py -n "+names[-1]+" -vect1 "+str(args.vect1[0])+" "+str(args.vect1[1])+" "+str(args.vect1[2])+" -vect2 "+str(args.vect2[0])+" "+str(args.vect2[1])+" "+str(args.vect2[2]) +" -vect3 "+str(args.vect3[0])+" "+str(args.vect3[1])+" "+str(args.vect3[2])+" -orientation "+str(args.orientation[0])+" "+ str(args.orientation[1])+" "+str(args.orientation[2]) +" -start_position "+str(args.start_position[0])+" "+str(args.start_position[1])+" "+str(args.start_position[2])+ " -normal_vector "+str(args.normal_vector[0])+" "+str(args.normal_vector[1])+" "+str(args.normal_vector[2])+" -distance "+str(args.distance)
		print(cmd)
		#destory the video stream window
		cv2.destroyAllWindows()
		#destory the video capture so that there is no timeout issue for next face detection
		vs.release()
		#run the command for writing name on board
		os.system(cmd)
		# start a new video capture
		vs=cv2.VideoCapture(gst)
		# start reacting frame by opening a new window for video stream
		ret,frame = vs.read()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

import cv2
import os
import face_recognition

# gstreamer plugins for linux for getting video frame from network
# resize the resolution for smoothness
gst="rtspsrc   location=rtsp://10.79.128.245/color   latency=5   !rtph264depay ! avdec_h264 ! videoscale!video/x-raw,width=480,height=270 ! videoconvert ! appsink "


# load the known faces and embeddings 
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("/home/dospina/Kinova_CV/encodings.pickle", "rb").read())


# Create the haar cascade for face detection from video stream
detector = cv2.CascadeClassifier('/home/dospina/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml')

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs =  cv2.VideoCapture(gst)

time.sleep(2.0)
# start the FPS counter
fps = FPS().start()
# loop over frames from the video file stream
while True:
    #capturing video using gstreamer plugins
	ret,frame = vs.read()

	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	# loop over the facial embeddings for face recognition 
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],encoding,0.55)
		name = "Unknown"
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

		# update the list of names
		names.append(name)
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
    # if the person is recognised the program that will write name will be called in the same terminal
	if len(names)!=0 and names[-1] != "Unknown":
        #command for running the program for writing name on board
		cmd="python3 /home/dospina/kortex/api_python/examples/102-Movement_high_level/Face_Recognition_Name.py -n "+names[-1]
        #destory the video stream window
		cv2.destroyAllWindows()
        #destory the video capture so that there is no timeout issue for next face detection
		vs.release()
        #run the command for writing name on board
		os.system(cmd)
        # start a new video capture
		vs=cv2.VideoCapture(gst)
        # start reacting frame by opening a new window for video stream
		ret,frame = vs.read()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
