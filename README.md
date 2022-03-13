# YOLOv3-tiny PyTorch implementation

## Features
* Train network
* Test network
* Inference on an image
* Inference on a video
* Inference on a camera movie

## Usage
### Train
```
python train.py [Arguments]
```

[Arguments (all are optional)]
* weights : pretrained weights used when transfer learning or fine tuning (default : None)
* model : specify 'sep' if you use separable model (default : None)
* data_root : path to dataset (train.txt must be placed, default : $HOME/datasets/COCO/2014)
* num_classes : num of classification
* class_names : path to file of class name (default : namefiles/coco.names)
* output_model : file name of trained model (default : yolo-tiny.pt)

* batchsize : batch size
* lr : initial value of learning rate
* epochs : training epochs
* decay : weights decay

* trans : transfer learning switch (default : False, set to True)
* finetune : fine tuning switch (default : False, set to True)
* novalid : don't do validation after every epoch (default : False, set to True)
* nosave : don't save training result (default : False, set to True)

Run train.py like above. This training program saves the trained model into 
'results/{date of end of training}/'. That directory includes model (*.pt), 
parameter list that have used in training (train_parameters.txt), 
learning rate transition (lr.png), mAP transition (mAP.png), 
and loss transition (loss.png).


### Test
```
python test.py [Arguments]
```

[Arguments (all are optional)]
* weights : trained model (default : weights/yolov3-tiny.pt)
* model : specify 'sep' if you use separable model (default : None)
* data_root : path to dataset (test.txt must be placed, default : $HOME/datasets/COCO/2014)
* num_classes : num of classification
* class_names : path to file of class name (default : namefiles/coco.names)
* quant : quantization switch (default : False, set to True)
* nogpu : specify if you don't want to use GPU (default : False, set to True)


### Inference
```
python detect.py [Arguments]
```

[Arguments (all are optional)]
* weights : trained model (default : weights/yolov3-tiny.pt)
* model : specify 'sep' if you use separable model (default : None)
* image : path to an image for inference (default : images/car.jpg)
* output_image : output file name of inference result (default : output.jpg) 
* num_classes : num of classification
* class_names : path to file of class name (default : namefiles/car.names)
* quant : quantization switch (default : False, set to True)
* nogpu : specify if you don't want to use GPU (default : False, set to True)

* conf_thres : confidence threshold for NMS
* nms_thres : nms threshold for NMS (IoU threshold)


### Note
* The directory 'tools/' is under development.
* Some codes are based on eriklindernoren/PyTorch-YOLOv3
