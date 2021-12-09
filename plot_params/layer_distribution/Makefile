#COPY_DIR=../pytorch_to_onnx_to_zybo
HEADER_DIR=bn_weight
COPY_DIR=../darknet_to_onnx_to_zybo/yaraki_output/

all: python

python:
	python view_dist.py $(HEADER_DIR)

copy:
	cp -f ${COPY_DIR}/conv*/conv_*_weight_bn.h bn_weight/.
