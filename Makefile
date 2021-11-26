all: debug

debug:
	rm -f params_debug/no-quant/* params_debug/quant/*
	python view_params.py --weights params_debug/yolov3-tiny_doll_light.pt --num_classes 2
	python view_params.py --weights params_debug/yolov3-tiny_doll_light_quant.pt --num_classes 2

