DETECT_OPTION = --num_classes 2 --class_names contest_2.names --conf_thres 0.1 --nogpu --quant
DETECT_IMAGE	= images/doll_light_1.png
WEIGHTS			  = weights/yolov3-tiny_doll_light_quant.pt

detect:
	python detect.py --weights $(WEIGHTS) $(DETECT_OPTION) --image $(DETECT_IMAGE)

