VIDEO_DIR	  := movies
VIDEO_FILE  := onsite_1.mp4 onsite_2.mp4
#VIDEO_FILES := $(addprefix $(VIDEO_DIR)/, $(VIDEO_FILE))
VIDEO  := $(VIDEO_DIR)/onsite_5.mp4
VIDEO1 := $(VIDEO_DIR)/onsite_1.mp4
VIDEO2 := $(VIDEO_DIR)/onsite_2.mp4
VIDEO3 := $(VIDEO_DIR)/onsite_3.mp4
VIDEO4 := $(VIDEO_DIR)/onsite_4.mp4
SAVE_DIR	  := $(addprefix images/, $(basename $(VIDEO_FILE)))

default: image

image:
	#python main.py $(VIDEO1) --savedir images/onsite_1
	#python main.py $(VIDEO2) --savedir images/onsite_2
	#python main.py $(VIDEO3) --savedir images/onsite_3
	#python main.py $(VIDEO4) --savedir images/onsite_4
	python main.py $(VIDEO) --savedir images/onsite_5
	#python main.py $(VIDEO_DIR)/gopro_1.mp4 --savedir images/gopro_1
	#python main.py $(VIDEO_DIR)/gopro_2.mp4 --savedir images/gopro_2
	#python main.py $(VIDEO_DIR)/gopro_3.mp4 --savedir images/gopro_3
	#python main.py $(VIDEO_DIR)/gopro_4.mp4 --savedir images/gopro_4
	#python main.py $(VIDEO_DIR)/gopro_5.mp4 --savedir images/gopro_5
	#python main.py $(VIDEO_DIR)/gopro_6.mp4 --savedir images/gopro_6
	#python main.py $(VIDEO_DIR)/gopro_7.mp4 --savedir images/gopro_7

demo:
	python main.py $(VIDEO2) --show --nosave

clean:
	rm -rf images/*

test:
	echo $(VIDEO_FILES)
