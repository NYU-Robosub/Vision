# robosubcv

## To train model:

### 1. Clone and Install YOLOv5 and Dependencies

$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt

## 2. Get and Prepare Data
We need to split our data into training, validation, and testing groups. A good split is 70% training, 15% validation, and 15% testing. 

## 3. Prepare the Training Configuration File
This should include the location of the training, validation, and testing data as well as the number of classes and class names.

## 4. Training the Model
We need to provide the image size, the number of batchs, the number of epochs and the path of the config file.

python train.py --img 640 --batch 32 --epochs 100 --data dataset.yaml --weights yolov5s.pt

When using the newly trained model to predict, we can provide the custom weights from training and the path to the new data.

python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source test.jpg

If we want the input to be from the webcam we change the source.

python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source 0
