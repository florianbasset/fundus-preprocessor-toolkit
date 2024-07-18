#read qnd save the image

import cv2
import os

def read_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found at {image_path}')
    else:
        return cv2.imread(image_path)

#ajout save image




