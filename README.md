# yolov8_practice

Image and video classification practice with yolov8 and OpenCV.  
<br>
**Frameworks**: DNN, Caffe, Darknet  
**Libraries/Modules**: numpy, OpenCV's config
**Algorithms**: YOLOv8  
<br>
> Note: Check Reference Videos and Model config Release

<br>
<div id="btt"></div>

## Contents
* [Object Detection Examples](#exs)
* [OpenCV Overview](#ocvOverview)
* [OpenCV's DNN](#ocvDNN)
* [DNN Process](#dnnProcess)
* [Resources](#resources)


<br>
<div id="exs"></div>

### Object Detection Examples
* I practiced with two images, one of a goose and another of super duper swaggalicious hard worker.
* First you'll see the original image, then the image with bounding box(es) after running the program.



<img src="https://github.com/jaszmine/yolov8_practice/assets/52623824/a46278dd-d89f-4ceb-a879-59322aa44dc3" alt="sillyGoose" style="float:right;width:500px;"/>
<br><br>
<img src="https://github.com/jaszmine/yolov8_practice/assets/52623824/dcb0530e-8781-4531-9c18-d1eecddb6c82" alt="run1" style="float:right;width:500px;"/>

<br><br><br>

<img src="https://github.com/jaszmine/yolov8_practice/assets/52623824/3f8a37d2-d5fa-4da1-9b56-4297934a4406" alt="stonks" style="float:right;width:500px;"/>
<br><br>
<img src="https://github.com/jaszmine/yolov8_practice/assets/52623824/5037d0ef-9d48-4d61-891d-ac7ef1420c52" alt="run2" style="float:right;width:500px;"/>




[Back to top](#btt)


<br>
<div id="ocvOverview"></div>

### OpenCV Overview
* An open-source computer vision and machine learning software library
* Applications
   * Facial recognition
   * Object identification
   * Human action classification
   * Camera movement tracking
* Natively written in C++, can use wrappers for Python and Java
* No framework-specific limitations
* An internal representation of models - can optimize code easier
* Has its own deep learning implementation - minimum external dependencies
* Uses BGR color format (instead of RGB)

[Back to top](#btt)

<br>
<div id="ocvDNN"></div>

### OpenCV's DNN
* Deep Neural Network Module
* NOT an entire deep learning framework
* Inference: 
   * When only a forward pass occurs (no back propgation so no default learning)
   * Engine example: input -> pretrained model -> result
   * Makes coding easier - no training means no GPUs needed
* OpenCV 4's DNN module supports:
   * Caffe
   * TensorFlow
   * Darknet
   * ONNX format 

[Back to top](#btt)

<br>
<div id="dnnProcess"></div>
   
### DNN Process
* Load pre-trained models from other DL frameworks
* Pre-process images using blobFromImages()
* Pass blobs through loaded pre-trained model to get output predictions (blob -> model -> inference)
* Read the Model
   * `cv2.dnn.readNetFromCaffe(protext, caffeModel)`
   * loads models and weights
* Create a Four-Dimensional Blob
   * `blob = cv2.dnn.blobFromImage(image, [scalefactor], [size], [mean], [swapRB], [crop], [ddepth])`
* Input the Blob into the Network
   * `net.setInput(blob)`
* Forward pass throught the Network
   * `outp = net.forward()`
   * produces an output prediction after a forward pass
* Summary of steps
1.  images
2.  blobFromImage()
3.  Blob
4.  Trained Model
5.  Inference
   * Returns: 4D Tensor(NCHW) - # of images, # of channels, height, width
   
|blobFromImage() Parameter| Description |
|:-:|:--|
|image| Input image (1, 3, or 4 channels) |
|scalefactor| Multiplier for image values |
|size| Spatial size for output image |
|mean| Scalar with mean values that are subtracted from BGR channels |
|swapRB| Flag that swaps channels (from RGB to BGR) |
|crop| Flag to crop image after resize |
|ddepth| Depth of ouput blob (CV_32F or CV_8U) |

[Back to top](#btt)


<br>
<div id="resources"></div>

### Resources
* [OpenCV DNN module docs](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
* [Deep Learning in OpenCV's GitHub Wiki](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV)
* [Conda dlib docs](https://anaconda.org/conda-forge/dlib)
* [Caffe docs](http://caffe.berkeleyvision.org/)
* [Darknet docs](https://pjreddie.com/darknet/)
