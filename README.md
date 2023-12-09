# Self-Driving Car Engineer Nanodegree
## Course 1: Computer Vision
From the Self-Driving Car Engineer Nanodegree programme offered at Udacity.

This is Course 1: Computer Vision in the Self-Driving Car Engineer Nanodegree programme

### Course Objectives
* Develop critical ML skills leveraged in autonomous vehicle engineering;
* Learn about the life-cycle of ML projects — from framing, selecting metrics, to training and model iteration;
* Learn to process raw digital images, perform camera calibration and distortion correction;
* Build convolutional neural networks using TensorFlow to classify and detect objects;
* Complete the Object Detection in an Urban Environment course project.


### Projects
* ✅ 1.1: [Object Detection in Urban Environments]


### Exercises
* ✅ 1.1.1: [Choosing Metrics]
* ✅ 1.1.2: [Data Acquisition and Visualisation]
* ✅ 1.1.3: [Creating TensorFlow TFRecords]
* ✅ 1.2.1: [Camera Calibration and Distortion Correction]
* ✅ 1.2.2: [Image Manipulation and Masking]
* ✅ 1.2.3: [Geometric Transformations]
* ✅ 1.3.1: [Logistic Regression]
* ✅ 1.3.2: [Stochastic Gradient Descent]
* ✅ 1.3.3: [Image Classification with Feedforward Neural Networks (FNNs)]
* ✅ 1.4.1: [Pooling Layers in Convolutional Neural Networks (CNNs)]
* ✅ 1.4.2: [Building Custom Convolutional Neural Networks (CNNs)]
* ✅ 1.4.3: [Image Augmentations for the Driving Domain]
* ✅ 1.5.1: [Non-Maximum Suppression (NMS) and Soft-NMS]
* ✅ 1.5.2: [Mean Average Precision (mAP)]
* ✅ 1.5.3: [Learning Rate Schedules and Adaptive Learning Rate methods]
* ✅ 1.6.1: [Fully Convolutional Networks]


### Course Contents

The following topics are covered in course exercises:
* Image classification with Convolutional Neural Networks (CNNs)
* Object detection with TensorFlow API
* Precision/recall, AP, mAP metrics for object detection
* Bounding box prediction
* Intersection over Union (IoU)
* Non-maximum Suppression / Soft-NMS
* Machine Learning (ML) workflows with TensorFlow Sequential, Functional API
* Model subclassing with TensorFlow
* Camera calibration (DLT, Levenberg-Marquardt)
* Camera pinhole and perspective projection models
* Recovering intrinsic/extrinsic parameters
* Colour thresholding
* Colour models (HSV, HSL, RGB)
* Binary masks and image masking
* Geometric transformations (affine, euclidean, etc.)
* Transformation and rotation matrices
* Data augmentation (e.g., random cropping, re-scaling, data generators, selective blending, etc.)
* Data augmentation with Albumentations (simulating motion, occlusions, time-of-day, etc.)
* Automated data augmentation (e.g., proxy/target tasks, policies, Smart/RandAugment, P/PBA)
* ETL pipelines
* Serialising binary datatypes (`.tfrecords` and `TFRecordDataset`)
* Protocol buffers (Google `protobuf`)
* Stochastic gradient descent
* Custom learning rate schedules
* Pooling and convolutional layers
* Padding and stride hyperparameters
* Filters and feature maps in 1D/2D
* Exploratory Data Analysis (EDA)
* TensorFlow Model Callbacks (e.g., TensorBoard)
* Image classification on MNIST and GTSRB datasets
* Traditional CNN architectures (LeNet-5)
* Tuning CNNs (e.g., addressing gradient decay, BatchNorm/Dropout, hyperparameter tuning, etc.)
* Building lightweight CNN architectures for embedded hardware
* Using custom activation functions (e.g., LeCun scaled `tanh`)
* Custom layer subclassing (e.g., Sub-sampling layer in LeNet-5)
* Selecting optimizers / loss and objective functions
* Complex model architectures and components (SSDs, RetinaNet, FPNs, RCNNs, SPPs)
* Improving object detection models for the self-driving car domain
* Monitoring GPU utilisation during training (and large-scale training on TPUs!)
* Designing skip connections;
* Transposed convolution layers;
* Fully Convolutional Networks and their performance (e.g., FPN-8);
* And so much more...


Other topics covered in course lectures and reading material:
* Deep learning history
* Tradeoffs
* Framing ML problems
* Metrics and error analysis in ML
* Economic impact and broader consequences of SDCs
* Camera models and calibration / reconstruction
* Pixel-level and geometric image transformations
* Backpropagation (and calculations performed by hand!)
* Traditional CNN architectures (AlexNet, VGG, ResNet, Inception)
* Selective search algorithm
* Region-proposal networks and improvements (RCNN, Fast-RCNN, Faster-RCNN, SPPNet)
* One- and two-stage detectors (YOLO versus SSD and CenterNet)
* Optimising deep neural networks (fine-tuning strategies, dropout / inverted dropout, batch normalisation)
* Transfer learning and applications at Waymo

### Learning Outcomes
#### Lesson 1: The Machine Learning Workflow
* Identify key stakeholders in a ML problem;
* Frame the ML problem;
* Perform exploratory data analysis (EDA) on an image dataset;
* Pick the most adequate model for a particular ML task;
* Choose the correct metric(s);
* Select and visualise the data.

#### Lesson 2: Sensor and Camera Calibration
* Manipulate image data;
* Calibrate an image using checkerboard images;
* Perform geometric transformation of an image;
* Perform pixel-level transformations of an image.

#### Lesson 3: From Linear Regression to Feedforward Neural Networks
* Implement a logistic regression model in TensorFlow;
* Implement back propagation;
* Implement gradient descent;
* Build a custom neural network for a classification task.

#### Lesson 4: Image Classification with Convolutional Neural Networks
* Write custom classification architectures using TensorFlow;
* Choose the right augmentations to increase dataset variability;
* Use regularisation techniques to prevent overfitting
* Calculate the output shape of a convolutional layer;
* Count the number of parameters in a convolutional network.

#### Lesson 5: Object Detection in Images
* Use the TensorFlow Object Detection API;
* Choose the best object detection model for a given problem;
* Optimise training processes to maximise resource usage;
* Implement Non-Maximum Suppression (NMS);
* Calculate Mean Average Precision (mAP);
* Choose hyper parameters to optimise a neural network.

#### Lesson 6: Fully Convolutional Networks
* Converting fully-connected to 1x1 convolution layers;
* Using transposed convolutions to upsample feature maps;
* Designing skip connections to improve segmentation map granularity;
* Encoder / decoder network architectures;
* Comparing the performance of fully convolutional networks (e.g., FCN-8s) to traditional CNNs;
* Implementing fully convolutional networks in TensorFlow using Sequential and Function API design patterns.

### Material
Syllabus:
* [Program Syllabus | Udacity Nanodegree](https://d20vrrgs8k4bvw.cloudfront.net/documents/en-US/Self-Driving+Car+Engineer+Nanodegree+Syllabus+nd0013+.pdf).

Literature:
* See specific assignments for related literature.

Datasets:
* [German Traffic Sign Recognition Benchmark](https://doi.org/10.17894/ucph.358970eb-0474-4d8f-90b5-3f124d9f9bc6) (GTSRB);
* [Waymo Open Dataset: Perception](https://waymo.com/open/).

### Other resources
Companion code:
* [Object Detection in an Urban Environment | Starter code](https://github.com/udacity/nd013-c1-vision-starter).
