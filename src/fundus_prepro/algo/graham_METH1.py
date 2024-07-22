import cv2
import numpy as np

def fundus_roi(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    threshold = 15
    _, roi = cv2.threshold(v, threshold, 1, cv2.THRESH_BINARY)
    roi = roi.astype(bool)
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

def load_ben_color(image, roi=None):
    sigmaX = 10
    image_roi = image.copy()
    image_roi[~roi] = 0 
    image_roi = cv2.addWeighted(image_roi, 4, cv2.GaussianBlur(image_roi, (0, 0), sigmaX), -4, 128)
    return {"image": image_roi}

def histogram_equalization_METH1(img, roi=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_roi = gray.copy()
    gray_roi[~roi] = 0  

    cdf = np.cumsum(np.histogram(gray_roi, 1024, [0, 1024])[0])
    cdf = cdf / cdf[-1]
    
    equalized_img = np.interp(gray, np.arange(0, 1024), cdf)
    equalized_img[~roi] = 0 
    equalized_img = np.array(equalized_img * 255, dtype='uint8')
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)
    
    return {"image": equalized_img}

def graham_meth1(image):
    data = fundus_roi(image)
    image_ben = load_ben_color(data["image"], data["roi"])
    final_image = histogram_equalization_METH1(image_ben["image"], data["roi"])
    return {"image": final_image}
