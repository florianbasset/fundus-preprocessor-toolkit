#plot images

import matplotlib.pyplot as plt
import cv2
import numpy as np

from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot

def cvt2bokeh(img):
    h, w = img.shape[:2]
    container = np.empty((h, w), dtype=np.uint32)
    view = container.view(dtype=np.uint8).reshape((h, w, 4))
    if img.ndim == 2: 
        view[:, :, 0] = img[::-1, :]  
        view[:, :, 1] = img[::-1, :]  
        view[:, :, 2] = img[::-1, :]  
    else: 
        view[:, :, 0] = img[::-1, :, 0] 
        view[:, :, 1] = img[::-1, :, 1] 
        view[:, :, 2] = img[::-1, :, 2] 
    view[:, :, 3] = 255
    return container

def plot_image(image, title='Image originale'):
    plt.imshow(image[:,:,::-1])
    plt.title(title)
    plt.axis('off')
    plt.show()

#librairie bokeh affichage image 
def plot_image_bokeh(image, title='Image originale'):
   image_original_bokeh = cvt2bokeh(image[:,:,::-1])
   output_notebook()
   if image_original_bokeh.shape[1] > 1800:
        p = figure(x_range=(0, image_original_bokeh.shape[1]),
                   y_range=(0, image_original_bokeh.shape[0]), 
                   width=image_original_bokeh.shape[1]//5, 
                   height=image_original_bokeh.shape[0]//5, 
                   title=title)
        p.image_rgba(image=[image_original_bokeh], x=0, y=0, dw=image_original_bokeh.shape[1], dh=image_original_bokeh.shape[0])
        p.title.text = title
        show(p)
   elif image_original_bokeh.shape[1] > 1000:
        p = figure(x_range=(0, image_original_bokeh.shape[1]), 
                   y_range=(0, image_original_bokeh.shape[0]), 
                   width=image_original_bokeh.shape[1]//4, 
                   height=image_original_bokeh.shape[0]//4, 
                   title=title)
        p.image_rgba(image=[image_original_bokeh], x=0, y=0, dw=image_original_bokeh.shape[1], dh=image_original_bokeh.shape[0])
        p.title.text = title
        show(p)
   elif image_original_bokeh.shape[1] > 500:
        p = figure(x_range=(0, image_original_bokeh.shape[1]), 
                   y_range=(0, image_original_bokeh.shape[0]), 
                   width=image_original_bokeh.shape[1]//3, 
                   height=image_original_bokeh.shape[0]//3, 
                   title=title)
        p.image_rgba(image=[image_original_bokeh], x=0, y=0, dw=image_original_bokeh.shape[1], dh=image_original_bokeh.shape[0])
        p.title.text = title
        show(p)
   else: 
        p.image_rgba(image=[image_original_bokeh], x=0, y=0, dw=image_original_bokeh.shape[1], dh=image_original_bokeh.shape[0])
        p.title.text = title
        show(p)
   