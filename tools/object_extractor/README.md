## Overview
Image extractor using NN object detection.
You can extract some images of vehicles from a video using this program.
That 'vehicles' includes car, bicycle, motorbike, bus, and truck, which are in COCO class.
This system extract an image according to the algorythm below.
1. Wait for some frames.
2. Run inferrence using YOLOv3 on the current frame.
3. If the frame is of some objects, get a difference image between the previous inferrence result and the current one, and mask the image to generate binary image.
4. If the amount of difference of the difference image exceeds the threshold, save the frame.
5. Repeat from 1 to 4 until the end of movie.

## Usage
'''
python main.py {path to source video}
'''

### Option (Arbitral)
- savedir {val} : path to the directory to save image (default : ./images) 
- interval {val} : num of frames to wait between YOLO inferrences (default : 120)
- mask_thres {val} : mask threshold to generate binary image (default : 150).
  The higher mask_thres is set, the less sensitive to small difference.
- save_thres {val} : sum of pixel value of difference image to save (default : 20000)
- show : switch to show binary image, detected image, and source movie (default : False, set to True)
- nosave : switch not to save image (default : False (i.e. save image), set to True)
