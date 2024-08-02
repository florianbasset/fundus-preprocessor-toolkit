import cv2
import numpy as np

from fundus_prepro.algo.graham_METH1 import fundus_roi


def load_ben_color(image, roi=None):
    sigmaX = 10
    image_roi = image.copy()
    image_roi[~roi] = 0 
    final_image = cv2.addWeighted(image_roi, 4, cv2.GaussianBlur(image_roi, (0, 0), sigmaX), -4, 128)
    return final_image  # Retourne directement l'image

def histogram_equalization_METH1(img, roi=None):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr_roi = img_bgr.copy()
    img_bgr_roi[~roi] = 0 

    ycrcb_img = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    equalized_img_bgr = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    final_image = cv2.cvtColor(equalized_img_bgr, cv2.COLOR_BGR2RGB)

    return final_image  # Retourne directement l'image

def graham_meth2(image):
    data = fundus_roi(image)
    image_ben = load_ben_color(data['image'], data["roi"])
    final_image = histogram_equalization_METH1(image_ben, data["roi"])
    return {"image": final_image}
