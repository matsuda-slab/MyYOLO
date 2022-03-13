#===============================================================================
# Load onnx model and do inferrence on tensorflow backend
#===============================================================================

import onnx_tf.backend
import onnx
import tensorflow as tf
from utils.transforms_detect import resize_aspect_ratio
import torch
import os
import sys
import argparse
import numpy as np
import random
import cv2
from utils.utils_tf import non_max_suppression
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/yolo-tiny.onnx')
parser.add_argument('--image', default='images/dog.jpg')
parser.add_argument('--conf_thres', type=float, default=0.5)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--output_image', default='output.jpg')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--class_names', default='coco.names')
parser.add_argument('--nogpu', action='store_true', default=False)
args = parser.parse_args()

weights_path = args.weights
image_path   = args.image
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
output_path  = args.output_image
NUM_CLASSES  = args.num_classes
name_file    = args.class_names
NO_GPU       = args.nogpu

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if NO_GPU:
#    device = torch.device("cpu")
#tensor_type = torch.cuda.FloatTensor
#                   if torch.cuda.is_available()
#                   else torch.FloatTensor

# Load class names from name file
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# Load model of onnx format
model = onnx.load(weights_path)

# Transform model to tensorflow format
tf_model = onnx_tf.backend.prepare(model, device='CPU')

start = time.time();

# Load an image
input_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

image_load_t = time.time()
input_image = input_image[:, :, :] / 255.0
image = resize_aspect_ratio(input_image, use_torch=False)      # by opencv
image = image.transpose(2, 0, 1)
image = image[np.newaxis, [2,1,0], :, :]
np.set_printoptions(edgeitems=2000)
image = tf.cast(image, tf.float32)
image_convert_t = time.time()

# Forward 
output = tf_model.run(image)       # output is in [0-1]

inference_t = time.time()

# NMS
output_torch = torch.from_numpy(output[0])
output = non_max_suppression(output_torch, conf_thres, nms_thres)
nms_boxes = output[0]
print("nms_boxes :", nms_boxes.shape)

# <!> nms() of tf is supposed to receive boxes of shape of [y1, x1, y2, x2]
#     note that output is [x1, y1, x2, y2, conf, cls]
#boxes  = output[0][0][:, 0:4]
#boxes[:, 0] = boxes[:, 1] / 416.0     # Normalization
#boxes[:, 1] = boxes[:, 0] / 416.0
#boxes[:, 2] = boxes[:, 3] / 416.0
#boxes[:, 3] = boxes[:, 2] / 416.0
#boxes  = boxes[:, [1,0,3,2]]
#scores = output[0][0][:, 4]
#selected_indices = tf.image.non_max_suppression(boxes, scores, 200,
#                                    iou_threshold=0.3, score_threshold=0.01)
#
nms_t = time.time()
#
##output = output[0]
#nms_boxes = tf.gather(boxes, selected_indices)
#nms_boxes = nms_boxes * 416.0

orig_h, orig_w = input_image.shape[0:2]
pad_x = max(orig_h - orig_w, 0) * (416 / max(orig_h, orig_w))
pad_y = max(orig_w - orig_h, 0) * (416 / max(orig_h, orig_w))

unpad_h = 416 - pad_y
unpad_w = 416 - pad_x

unpad_boxes = np.zeros((nms_boxes.shape[0], 4))
unpad_boxes[:, 0] = ((nms_boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
unpad_boxes[:, 1] = ((nms_boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
unpad_boxes[:, 2] = ((nms_boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
unpad_boxes[:, 3] = ((nms_boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

#plt.figure()
#fig, ax = plt.subplots(1)
#ax.imshow(rgb_image)

# Create color map
#cmap = plt.get_cmap('tab20b')
#colors = [cmap(i) for i in np.linspace(0, 1, NUM_CLASSES)]
#bbox_colors = random.sample(colors, NUM_CLASSES)

### Create rectangle and label
for x_min, y_min, x_max, y_max in unpad_boxes:
    box_w = x_max - x_min
    box_h = y_max - y_min

    #color = bbox_colors[int(class_pred)]
    #bbox = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2,
    #                            edgecolor=color, facecolor='None')
    #ax.add_patch(bbox)

    # label
    #plt.text(x_min, y_min, s=class_names[int(class_pred)], color='white',
    #               verticalalignment='top', bbox={'color': color, 'pad':0})
    cv2.rectangle(input_image, (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)), (0, 0, 255), thickness=2)
    cv2.putText(input_image, 'doll', (int(x_min), int(y_min)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 255), thickness=2)

end = time.time()
print("elapsed time = %.4f sec" % (end - start))
print("items :")
print(" image_load : %.4f sec" % (image_load_t - start))
print(" image_convert : %.4f sec" % (image_convert_t - image_load_t))
print(" inference : %.4f sec" % (inference_t - image_convert_t))
print(" nms : %.4f sec" % (nms_t - inference_t))
print(" plot : %.4f sec" % (end - nms_t))

#plt.axis("off")
#plt.savefig(output_path)
#plt.close()
cv2.imshow('image', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
