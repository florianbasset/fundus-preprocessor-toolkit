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
    final_image = cv2.addWeighted(image_roi, 4, cv2.GaussianBlur(image_roi, (0, 0), sigmaX), -4, 128)
    return {"image": final_image}

def histogram_equalization_METH2(img, roi=None):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr_roi = img_bgr.copy()
    img_bgr_roi[~roi] = 0 

    ycrcb_img = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    equalized_img_bgr = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    final_image = cv2.cvtColor(equalized_img_bgr, cv2.COLOR_BGR2RGB)

    return {"image": final_image}

def graham_meth2(image):
    data = fundus_roi(image)
    image_ben = load_ben_color(data['image'], data["roi"])
    final_image = histogram_equalization_METH2(image_ben["image"], data["roi"])
    return {"image": final_image}
