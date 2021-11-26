NUM_CLASSES	  = 4
CLASS_FILE		= google_cars.names
DETECT_OPTION = --num_classes $(NUM_CLASSES) --class_names $(CLASS_FILE) --conf_thres 0.1 --nogpu --quant
DETECT_IMAGE	= images/doll_light_1.png
DETECT_VIDEO	= images/car.mp4
WEIGHTS			  = weights/yolov3-tiny_car4_quant_jit.pt

detect:
	python detect.py --weights $(WEIGHTS) $(DETECT_OPTION) --image $(DETECT_IMAGE)

video:
	python detect_video.py --weights $(WEIGHTS) $(DETECT_OPTION) --video $(DETECT_VIDEO)
