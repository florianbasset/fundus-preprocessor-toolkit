import cv2
import numpy as np


def fundus_roi(image, mask=None):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    threshold = 15
    _, roi = cv2.threshold(v, threshold, 1, cv2.THRESH_BINARY)
    roi = roi.astype(np.uint8)
    white_pixels = np.argwhere(roi == 1)
    if white_pixels.size == 0:
        print("Aucun pixel blanc trouvé dans le masque.")
        return {"roi": roi, "diameter": 0, "image": image}
    x_min, y_min = np.min(white_pixels, axis=0)
    x_max, y_max = np.max(white_pixels, axis=0)
    diameter_x = x_max - x_min
    diameter_y = y_max - y_min
    diameter = int(np.maximum(diameter_x, diameter_y))
    return {"roi": roi, "diameter": diameter, "image": image}

def apply_clahe(image, diameter=None, roi=None):
    #if diameter is None:
    #    print('Erreur: Le diamètre est None. Impossible de continuer.')
    #    return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:  # Grayscale image
        return {"image": clahe.apply(image), "diameter": diameter}
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_planes = list(cv2.split(hsv_image))  # Convert to list to allow modification
        hsv_planes[2] = clahe.apply(hsv_planes[2])
        return {"image": cv2.cvtColor(cv2.merge(hsv_planes), cv2.COLOR_HSV2BGR), "diameter": diameter}

def sarki(image, diameter=None, roi=None):
    data = fundus_roi(image)
    clahe_image = apply_clahe(**data)

    hsv_clahe_image = cv2.cvtColor(clahe_image["image"], cv2.COLOR_BGR2HSV)
    hsv_float = hsv_clahe_image.astype(np.float32) / 255.0
    mu0 = 0.5  # Intensity target
    d1 = int(clahe_image["diameter"] / 15)
    if d1 % 2 == 0:
        d1 += 1
    muL = cv2.blur(hsv_float[:, :, 2], (d1, d1))  # Local mean

    # Processing pixel value
    v_processed = hsv_float[:, :, 2] + mu0 - muL
    v_processed = np.clip(v_processed, 0, 1)
    hsv_float[:, :, 2] = v_processed
    
    hsv_uint8 = (hsv_float * 255).astype(np.uint8)
    final_image = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)

    return {"image":final_image}