
import numpy as np

def autobalance(image):
    final_image = np.zeros_like(image)
    for channel_index in range(3):
        # Calcul de l'histogramme
        hist, bins = np.histogram(image[..., channel_index].ravel(), 256, (0, 256))
        # Calcul de la valeur min et max
        bmin = np.min(np.where(hist > (hist.sum() * 0.0005)))
        bmax = np.max(np.where(hist > (hist.sum() * 0.0005)))
        # Appliquer la transformation
        final_image[..., channel_index] = np.clip(image[..., channel_index], bmin, bmax)
        # Normaliser
        final_image[..., channel_index] = ((final_image[..., channel_index] - bmin) / (bmax - bmin) * 255)
    
    return {'image': final_image}

