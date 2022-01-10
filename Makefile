NUM_CLASSES	  = 80
CLASS_FILE		= namefiles/coco.names
CONF_THRES		= 0.01
DETECT_OPTION = --num_classes $(NUM_CLASSES) --class_names $(CLASS_FILE) --conf_thres $(CONF_THRES)
DETECT_IMAGE	= images/doll_light_1.png
DETECT_VIDEO	= images/car.mp4
WEIGHTS			  = weights/yolov3-tiny.pt
DATASET				= $(HOME)/datasets/COCO/2014

detect:
	python detect.py --weights $(WEIGHTS) $(DETECT_OPTION) \
									 --image $(DETECT_IMAGE)

test:
	python test.py --weights $(WEIGHTS) $(DETECT_OPTION) \
								 --data_root $(DATASET)

video:
	python detect_video.py --weights $(WEIGHTS) $(DETECT_OPTION) \
												 --video $(DETECT_VIDEO)

debug:
	rm -f params_debug/no-quant/* params_debug/quant/*
	python view_params.py --weights params_debug/yolov3-tiny_doll_light.pt \
												--num_classes 2
	python view_params.py --weights params_debug/yolov3-tiny_doll_light_quant.pt \
												--num_classes 2

distrib:
	python test.py --weights $(WEIGHTS) $(DETECT_OPTION) \
								 --data_root $(DATASET) --distrib
