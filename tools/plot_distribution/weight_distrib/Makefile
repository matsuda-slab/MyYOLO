WEIGHTS_PATH		= weights/no-merge/yolov3-tiny_car1_sep
WEIGHTS_PATH_NP = $(subst no-merge,merge, $(WEIGHTS_PATH))
TARGET					= $(WEIGHTS_PATH)/$(addsuffix .pt,$(notdir $(WEIGHTS_PATH)))
TARGET_NP				= $(WEIGHTS_PATH_NP)/$(addsuffix .npy,$(notdir $(WEIGHTS_PATH_NP)))
NUM_CLASSES			= 1
IMAGE						= weight_distribution.png

default: plot

plot:
	python plot_distrib.py --weights $(TARGET) --savedir $(WEIGHTS_PATH) \
												 --mode torch --num_classes $(NUM_CLASSES)

plot_np:
	python plot_distrib.py --weights $(TARGET_NP) --savedir $(WEIGHTS_PATH_NP) \
												 --mode numpy --num_classes $(NUM_CLASSES)

view:
	eog $(WEIGHTS_PATH_NP)/$(IMAGE) &

clean:
	rm -f *.png

echo:
	echo $(WEIGHTS_PATH_NP)
