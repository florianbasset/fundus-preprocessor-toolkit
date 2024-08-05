import cv2
import numpy as np

def clahe_max_green_gsc(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    green_channel = image[:, :, 1]
    max_pixel_image = np.maximum.reduce([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_clahe = clahe.apply(gray_image)
    green_clahe = clahe.apply(green_channel)
    max_pixel_clahe = clahe.apply(max_pixel_image)
    final_image = cv2.merge([gray_clahe, green_clahe, max_pixel_clahe])
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    return {"image":final_image}