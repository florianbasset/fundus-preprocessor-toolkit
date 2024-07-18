import cv2

def clahe_rgb(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image)
    cl_channels = [clahe.apply(channel) for channel in channels]
    final_image = cv2.merge(cl_channels)
    return {"image":final_image}
