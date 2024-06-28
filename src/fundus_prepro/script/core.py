#contient la fonction principale

from fundus_prepro.script.preprocessor import Preprocessor   

def preprocess_image(image_path, color_space, method):
    preprocessor = Preprocessor(image_path)
    image = preprocessor.apply_preprocessing(color_space, method)
    return image

