#contient la fonction principale

from fundus_prepro.script.preprocessor import Preprocessor   

def preprocess_image(image, color_space, method):
    preprocessor = Preprocessor(image)
    image = preprocessor.apply_preprocessing(color_space, method)
    return image

