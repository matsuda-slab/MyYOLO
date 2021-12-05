import cv2
import numpy as np

def resize_aspect_ratio(input_image, desired_size=416, use_torch=True):
    #image = cv2.imread(image_path)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    if use_torch:
        image = input_image.detach().cpu().numpy()
        image = image.transpose(1,2,0)
    else:
        image = input_image

    old_size = image.shape[:2]      # (height, width)

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=color)

    #cv2.imshow("image", new_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return new_image
