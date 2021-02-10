"""
This is Old JSW project we need integrate object tracking time when
object reach a certain limit inside the plant.
"""
# Importing libraries

import threading
import uuid
import logging
import logging.handlers as handlers
from datetime import datetime
import os
import cv2, time
import numpy as nppbtxt
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
# pymssql
from object_detection import utils
#from object_detection.utils import visualization_utils as vis_util
from threading import Thread
import pandas as pd
# This is needed since the notebook is stored in the object_detection folder.
#import value as value

np.set_printoptions(precision=2)
sys.path.append("..")

# Import utilites
# from utils import label_map_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# from scipy.spatial import distance as dist
# get image and display (opencv)
log = logging.getLogger('opencv_save_multi_images3')
log.setLevel(logging.DEBUG)
timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
u_id = str(uuid.uuid4().hex.upper()[0:6])
log_file = "/mnt/a/logs/testing/opencv_save_multi_images_{}_{}.log".format(timestr, u_id)
# fh = logging.FileHandler(log_file)
# fh = handlers.TimedRotatingFileHandler(log_file, when='D', interval=1)
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
# fh.setFormatter(formatter)
# log.addHandler(fh)

import logging
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# LOGGER
l1 = datetime.now()
logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)
# add a time rotating handler
handler = TimedRotatingFileHandler("/home/vert/Desktop/log/logger_{}.log".format(l1), when="m",interval=60)
logger.addHandler(handler)

# centroid function
def pega_centro(l, r, t, b):
    x1 = int(t / 2)
    y1 = int(b / 2)
    cx = l + x1
    cy = r + y1
    return cx,cy
'''
Centroid Calculation:
L - From image one side edge to left edge of a detected object.
R – From image one side edge to last right edge of a detected object.
T – From top of the image to first top edge of detected object.
B – From top of the image to bottom edge of detected object.
If it asking int required just add int before formula

Centroid => C1 = (L+(R-L)/2)
	        C2 = (T+(T-B)/2)
cv2.circle(img,(c1,c2),1,7,5)

'''

# This the main function to perform all
def camera_connection(process):

    inside=0
    print("Started process", process)
    # log.info("{}:Started process".format(process))

    # Taking current directory
    CWD_PATH = os.getcwd()
    # warning_enable = False
    try:
        # General name for the model
        MODEL_NAME = 'inference_graph1'

        # Assigning label txt file
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'training1', 'labelmap.pbtxt')


        # PATH_TO_LABELS="/home/server3/tensorflow1/models/research/object_detection/shell2sideviewtraining/labelmap.pbtxt"
        NUM_CLASSES = 6

        # Path to frozen detection graph .pb file, which contains the model that is used for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

        # print("label_map={}".format(label_map))
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        # Taking Categories
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # with tf.io.GFile(PATH_TO_CKPT, 'rb') as fid:
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # This is tensor flow 1.x code to run the above model
            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        # Class detection
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # print("Biswa", num_detections)

        # Reading local video
        cap = cv2.VideoCapture('/home/vert/Foreign_Obj_Det_Project/Project_Demo_Video/Shell2FrontView_wop_20210209_140215007565.avi')

        # Running while loop into the video to testing the code
        while True:

            _,img = cap.read()
            img1=img.copy()

            # Scaling down the image to display
            scale_percent1 = 40  # percent of original size
            width1 = int(img1.shape[1] * scale_percent1 / 100)
            height1 = int(img1.shape[0] * scale_percent1 / 100)
            dim1 = (width1, height1)
            nf1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)
            # vout1.write(nf1)
            frame_expanded = np.expand_dims(img, axis=0)

            # Copy image the image
            # img=nf1.copy()

            # DRAWING LINE for shell2 side view
            # Left LINE
            x1, y1 = 800, 950
            x1, y2 = 600, 1050
            # off1=2.0
            cv2.line(img, (x1, y1), (x1, y2), (0, 0, 255), 2)

            # Right LINE
            X1, Y1 = 1200, 950
            X1, Y2 =1200, 1050
            # off2=2.0275
            cv2.line(img, (X1, Y1), (X1, Y2), (0, 0, 255), 2)

            # # LINE - 3
            p1, q1 = 600, 750
            p2, q1 = 1150, 950
            # off3 = 10
            cv2.line(img, (p1, q1), (p2, q1), (0, 0, 255), 2)
            #
            # LINE - 4
            P1, Q1 = 600, 850
            P2, Q1 = 1150, 1100
            # off3 = 10
            cv2.line(img, (P1, Q1), (P2, Q1), (0, 0, 255), 2)
            # #
            #
            # # DRAWING LINE for shell1 front view
            #
            # # Right LINE-1
            # x1, y1 = 700, 800
            # x1, y2 = 600, 950
            # # off1=2.0
            # # cv2.line(img, (x1, y1), (x1, y2), (0, 0, 255), 2)
            #
            # # Left LINE-2
            # X1, Y1 = 1200, 800
            # X1, Y2 = 1200, 950
            # # off2=2.0
            # # cv2.line(img, (X1, Y1), (X1, Y2), (0, 0, 255), 2)
            #
            # # LINE - 3
            # p1, q1 = 600, 750
            # p2, q1 = 1200, 800
            # # off3 = 10
            # # cv2.line(img, (p1, q1), (p2, q1), (0, 0, 255), 2)
            #
            # # LINE - 4
            # P1, Q1 = 600, 850
            # P2, Q1 = 1200, 950
            # off3 = 10
            # cv2.line(img, (P1, Q1), (P2, Q1), (0, 0, 255), 2)
            # print("Empty class found end")

            # print("{} shape {}".format(process,frame_expanded.shape))
            #log.info("{} shape {}".format(process, frame_expanded.shape))

            # Assighing the boxes, scores, class and num from the model
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            # Converting image to array to pass inside the model
            img1 = np.array(img1)

            # Taking coordinates from function which is available in util folder
            im,ymin,xmin,ymax,xmax=vis_util.visualize_boxes_and_labels_on_image_array(img1,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.40)

            # Taking boxes details
            boxes1 = np.squeeze(boxes)

            # get all boxes from an array
            max_boxes_to_draw = boxes1.shape[0]

            # get scores to get a threshold
            scores1 = np.squeeze(scores)
            # print(scores1)

            # this is set as a default but feel free to adjust it to your needs
            min_score_thresh = .4

            # Checking person class from the video to take the time
            for i in range(min(max_boxes_to_draw, boxes1.shape[0])):
                if scores1 is None or scores1[i] > min_score_thresh:
                    class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name']

                    # If the object is person then we need to process below lines
                    if class_name=="Person":
                        # print(class_name)

                        # Taking the shape of the image
                        h,w = img.shape[:2]

                        # Taking l,r,t,b value from the box
                        l,r,t,b=int(xmin*w),int(xmax*w),int(ymin*h),int(ymax*h)

                        # Printing logger
                        logger.info("print l,r,t,b-{}-{}-{}-{}".format (l,r,t,b))
                        # print("Coordinates-{}-{}-{}-{}".format(l,r,t,b))

                        # Creating Rectangle box out side of the object
                        cv2.rectangle(img,(l,b),(r,t),(1,190,200),2)
                        logger.info("print image shape-{}".img.shape())

                        # Creating bounding box with different color
                        # cv2.rectangle(img, (l, b), (r, t),(0,0,255), 1)

                        # centroid calculation
                        c1 = int(l+((r-l)/2))
                        c2 = int(t+((b-t)/2))
                        logger.info("print ci and c2 -{}-{}".format (c1,c2))

                        # Applying cv2 circle method to create centroid in side the object
                        cv2.circle(img,(c1,c2),1,(255,153,255),3)
                        cv2.putText(img,"Person",(l,t),cv2.FONT_HERSHEY_PLAIN,1,(255,127,0),1)

                        # Checking the distance to take the time
                        # if q1<c2+offset and q1>c2-offset:
                        #     print("Person Crossed The Line")
                        #     # Writing the current timestamp inside the txt file in our folder
                        #     with open("/home/vert/Desktop/Time_Tracking/Log.txt","a") as f:
                        #         f.write("Person Crossed Line-{}".format(datetime.now()))
                        #         f.write("\n")

                        # Checking the condition to take the time
                        if q1 < c2 < Q1 and x1 < c1 < X1:
                            inside += 1
                            if inside == 1:

                                # He is inside the warning zone
                                print("INSIDE WARNING ZONE")
                                with open("/home/vert/Desktop/Time_Tracking/Log.txt", "a") as f:
                                    f.write("Person Crossed Line-{}".format(datetime.now()))
                                    f.write("\n")
                                cv2.putText(img, "Inside Warning zone", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        else:
                            # if (inside>=1 and c2<q1) or (inside>=1 and c2>Q1):
                            if inside > 0:
                                # HE's OUT
                                print("OUTSIDE WARNING ZONE")
                                with open("/home/vert/Desktop/Time_Tracking/Log.txt", "a") as f:
                                    f.write("Outside Warning Zone-{}".format(datetime.now()))
                                    f.write("\n")
                                cv2.putText(img, "Out of Warning Zone", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (130, 255, 255), 2)
                                # RESET inside to "0"
                                inside=0
                else:
                    pass
            cv2.imshow("SSD Model Image",nf1)
            cv2.imshow("Normal Image",img)
            cv2.waitKey(1)
            # if cv2.waitKey(1) == 27:
            #   break

    except (Exception, Exception) as exc:
        print("Ending Processing..theres exceptiPon-{}".format(exc))

# if __name__ == '__main__':
camera_connection("t1")
