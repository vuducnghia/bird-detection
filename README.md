# BIRD DETECTION

## About
- Purpose: Detect and locating birds on images captured by edge devices.

## Technology
- use [YOLOv8n](https://github.com/ultralytics/ultralytics)( SOTA for realtime object detection)
- Python >= 3.8
- Deep learning
- Quantization: INT8

## How does it work?
YOLOv8 can detect objects by using a single convolutional network that divides the input image into a grid of cells and predicts bounding boxes and probabilities for each cell. 
It also uses a novel attention mechanism to focus on the most relevant features of the objects. 

YOLOv8 is an algorithm that uses convolutional neural networks (CNN) to detect objects in real-time. 
CNN is a type of deep learning model that can learn features from images and perform various tasks such as classification and detection. 
YOLOv8 uses CNN to detect objects because of the following reasons:
- CNN can extract high-level features from images, such as edges, shapes, colors, and textures, which are useful for identifying objects.
- CNN can handle complex and varied inputs, such as different sizes, orientations, and lighting conditions of objects, which are common in real-world scenarios.
- CNN can perform end-to-end learning, which means that it can learn the mapping from raw pixels to object labels and bounding boxes without any intermediate steps or handcrafted features.
- CNN can achieve high accuracy and speed in object detection, which are the main goals of YOLOv8.

## Whaf I have implemented:
1, Collect data on birds captured by drones. [example](test/000703.png) <br />
2, Label the [data](results/label_image.png). <br />
3, Check data [labels](results/labels.jpg), [labels correlogram](results/labels_correlogram.jpg)<br />
5, Config param to training [params](results/args.yaml)
4, Training data( 2.046 images for training + 395 images for validation) [train_batch0](results/train_batch0.jpg), [train_batch1](results/train_batch1.jpg), [train_batch2](results/train_batch2.jpg), ...
[train_batch2880](results/train_batch2880.jpg), [train_batch2881](results/train_batch2881.jpg), [train_batch2882](results/train_batch2882.jpg)<br />
5, Check [result](results/results.csv) and [loss](results/loss.png)<br />
6, Choose best model

## What is left to do
- Deploy practical models and evaluate effectiveness.
- Collect more real-world data to make the model more accurate.
- Optimize performance and accuracy.

## Install
```
pip install -r requirements.txt
```

## Predict and Use
Depending on the speed you want, I offer you 3 models:
- best8n.pt: This is the original model after training.
- best8n.onnx: This is the model that has been converted to onnx format. The size of this model is heavier but in return its speed is faster( on CPU) than the original model with unchanged accuracy.
- best8n_quant.onnx: This is a quantized model so the speed is the fastest but the accuracy will decrease by a few percent.
- There is always a trade-off between speed and accuracy. You can flexibly choose to suit your needs.

Change path to model in: 
``
model = YOLO("results/weights/best.pt")
``

Run predict: 
```
python predict.py
```
Arguments:
- source: source images.
- imgsz: image size as scalar(640 or 480). The 480 runs faster but has reduced accuracy.
- conf: object confidence threshold for detection. 

``model.predict(source="test/images/000058.png", imgsz=640, conf=0.45)``

## Model visualizer
[model visualizer](model_visualizer.png)
