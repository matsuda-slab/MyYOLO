# Overview
Generate graph of parameter distribution.
Output distribution of parameters to png file, and output distribution of value
over all layer to standard output.

# Usage
Specify the weight path (*.pt or *.npy), and this program save the graph to the
same directory as the specified weight file.
If you use *.pt file for weight file, firstly edit 'WEIGHTS_PATH' in Makefile to the
weight file path. Note that the directory name must be same as the weight name 
(ex. yolov3-tiny/yolov3-tiny.pt).
And then run `make plot`.
If you use *.npy file for weight file, edit 'WEIGHTS_PATH_NP' in Makefile, and
run `make plot_np`.
