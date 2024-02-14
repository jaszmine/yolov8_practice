import numpy as np
import cv2
from ultralytics import YOLO
import random

# open file in read mode
my_file = open("utils/coco.txt", "r")
# reading file
data = my_file.read()
# replacing end, splitting text | when newline ('\n') is seen
class_list = data.split("\n")
my_file.close()

#print(class_list)

# generate rand colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained yolov8n model
model = YOLO("weights/yolov8n.pt", "v8")

# vals to resize video frames | small fram optimise the run
frame_wid = 640
frame_hyt = 480

# cap = cv2.VideoCaputre(1)
cap = cv2. VideoCapture("italians!!.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame's read correctly, ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # resize frame | small frame optimize the run
    #frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # write frame to be loaded by the prediction module
    cv2.imwrite("stonks.jpg", frame) 

    # predict on image
    detect_params = model.predict(source="stonks.jpg", conf=0.45, save=False)

    # convert tensor array to numpy
    print(detect_params[0].numpy())
    detect_params = detect_params[0].numpy()

    if len(detect_params) !=0 :

        # loop through all detections in current frame
        for param in detect_params:
            #print(param[1])

            # draw bbox around detection
            cv2.rectangle(frame, (int(param[0]),int(param[1])), 
                                 (int(param[2]),int(param[3])), 
                                 detection_colors[int(param[5])], 3)
            
            # display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, class_list[int(param[5])] +
                        " " + str(round(param[4], 3)) + "%", 
                        (int(param[0]),
                         int(param[1])-10),
                         font, 1, (255,255,255),2)
            
    # display the resulting frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        break

    # when everything's done, release the capture
    #cap.release()
    #cv2.destroyAllWindows()
