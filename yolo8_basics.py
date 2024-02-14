from ultralytics import YOLO
import numpy

# load a pretrained yoloV8 model
model = YOLO("yolov8n.pt", "v8")

# predict on an image
#detection_ouput = model.predict(source="sillyGoose.jpg", conf=0.25, save=True)
detection_ouput = model.predict(source="stonks.jpg", conf=0.25, save=True)

# display tensor array
print(detection_ouput)

# display numpy array
print(detection_ouput[0].numpy())