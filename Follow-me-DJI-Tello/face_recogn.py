import tensorflow as tf

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from matplotlib.patches import Rectangle
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from cv2 import cv2
import os

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction= 0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
detector = MTCNN()

import cv2
from djitellopy import tello

me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()


def face_detect(img):
    """
    This function takes the path of an image as input.
    The image is run through the MTCNN face detector; the detected
    faces are then run through our neural network to detect the
    presence of a face mask and the result is given as an
    output image.
    """
    # print(img.shape)
    image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    w, h, _ = image.shape
    if w < 224 and h < 224:
        print("Image resolution too low to be analyzed!")
        return

    locs = []
  

    # pass the image through the MTCNN and obtain the face detections
    faces = detector.detect_faces(image)
    
    face_counter = 0
    # loop over the detection

    for face in faces:

        # extract the confidence associated with the detection
        confidence = face["confidence"]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.9:
            face_counter += 1
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = face["box"]
            (startX, startY, width, height) = box
            
            locs.append((startX, startY, startX+width, startY+height))

    return locs

model = load_model("MODEL.CLEAN_V3.model.weights.hdf5")






while True:

    img = me.get_frame_read().frame
    img = imutils.resize(img, width=400)
    locs = face_detect(img)


    for box in locs:

	    (startX, startY, endX, endY) = box

	    color = (0, 255, 0) 
	    cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)


    img = cv2.resize(img, (360, 240))

    cv2.imshow("lol", img)

    cv2.waitKey(1)