#plot images

import matplotlib.pyplot as plt
import cv2
import holoviews as hv
hv.extension('bokeh')

def plot_image(image, title='Image originale'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def hv_plot_image(image, title='Image originale'):
    img = hv.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bounds=(0, 0, image.shape[1], image.shape[0]))
    img.opts(title=title, width=image.shape[1], height=image.shape[0], tools=['hover'], colorbar=True)
    return img