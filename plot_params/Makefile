TARGET=weights/yolov3-tiny_car1.pt
NUM_CLASSES=1
PLOTFILE=plot.plt
IMAGE=params_distri.png

default: txt plot

txt:
	python plot_distrib.py --weights $(TARGET) --num_classes $(NUM_CLASSES)

plot:
	gnuplot $(PLOTFILE)

view:
	eog $(IMAGE) &
