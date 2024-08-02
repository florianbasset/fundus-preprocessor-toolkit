import cv2
import numpy as np

from fundus_prepro.algo.graham_METH1 import fundus_roi

def clahe_hsv(image, diameter=None, roi=None):
    #if diameter is None:
    #    print('Erreur: Le diamètre est None. Impossible de continuer.')
    #    return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:  # Grayscale image
        return {"image": clahe.apply(image), "diameter": diameter}
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_planes = list(cv2.split(hsv_image))  # Convert to list to allow modification
        hsv_planes[2] = clahe.apply(hsv_planes[2])
        return {"image": cv2.cvtColor(cv2.merge(hsv_planes), cv2.COLOR_HSV2RGB), "diameter": diameter}

def sarki(image, diameter=None, mask=None):
    data = fundus_roi(image)
    clahe_image = clahe_hsv(**data)

    hsv_clahe_image = cv2.cvtColor(clahe_image["image"], cv2.COLOR_RGB2HSV)
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
    final_image = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)

    # Apply ROI mask
    roi = data["roi"]
    final_image[~roi] = 0  # Set pixels outside ROI to black

    return {"image":final_image}


