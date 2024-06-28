#read qnd save the image

import cv2
import os

def read_image(image_path):
    if not os.path.exists(image_path):
        print("the file doesn't exist")
    else:
        print('The file exists')
        return cv2.imread(image_path)

def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    cv2.imwrite(output_path, image)

