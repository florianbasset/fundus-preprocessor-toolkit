import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

class Preprocessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)/255.0
        self.image = self.image.astype(np.float32)

    
    def to_hsv(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
    
    def to_lab(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
    
    #def to_bgr(self):
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            if np.array_equal(self.image, self.to_hsv()):
                return cv2.cvtColor(self.image, cv2.COLOR_HSV2BGR)  
            elif np.array_equal(self.image, self.to_lab()):
                cv2.cvtColor(self.image, cv2.COLOR_LAB2BGR)
        else :
            return self.image
    
    def apply_operation(self, image, operation, color_space):
        if color_space == 'HSV':
            image = self.to_hsv()
        elif color_space == 'LAB':
            image = self.to_lab()

        if operation == 'CLAHE':
            a = self.apply_clahe(image)
        elif operation == 'Seoud':
            a = self.apply_seoud(image)
        elif operation == 'illumination':
            a = self.illumination_equalization(image,1000)
        elif operation == 'contrast_equalizer':
            a = self.adaptive_contrast_equalization(image, 1000)
        elif operation == 'denoising':
            a = self.denoising(a)
        elif operation == 'Sarki':
            a = self.apply_sarki(image)
        elif operation == 'Bilateral':
            a = self.apply_bilateral_filter(image)
        elif operation == 'Tophat':
            a = self.apply_tophat(image)
        elif operation == 'Blackhat':
            a = self.apply_balckhat(image)
        elif operation == 'Gaussian':
            a =  self.apply_gaussian_blur(image)
        elif operation == 'IntensityNorm':
            a = self.apply_intensity_normalization(image)
        elif operation == 'combinaison1':
            a = self.apply_combinaison1(image) 
        elif operation == 'combinaison2':
            a =  self.apply_combinaison2(image) 
        elif operation == 'combinaison3':
            a =  self.apply_combinaison3(image) 
        elif operation == 'combinaison4':
            a =  self.apply_combinaison4(image) 
        elif operation == 'combinaison5':
            a =  self.apply_combinaison5(image) 
        else:
            raise ValueError("Unsupported operation")
        
        if color_space == 'LAB': 
            return cv2.cvtColor(a,cv2.COLOR_LAB2BGR)
        elif color_space == 'HSV':
            return cv2.cvtColor(a,cv2.COLOR_HSV2BGR)
        elif color_space == 'RGB':
            return a
        
    def apply_clahe(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 2:  # Grayscale image
            return clahe.apply(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:  #color image
            if np.array_equal(image, self.to_hsv()):  # HSV
                hsv_planes = cv2.split(image)
                hsv_planes_list = list(hsv_planes)
                hsv_planes_list[2] = clahe.apply(hsv_planes_list[2])
                return cv2.merge(hsv_planes_list)
            elif np.array_equal(image, self.to_lab()):  # LAB
                lab_planes = cv2.split(image)
                lab_planes_list = list(lab_planes)
                lab_planes_list[0] = clahe.apply(lab_planes_list[0])
                return cv2.merge(tuple(lab_planes_list))
        raise ValueError("Unsupported image format for CLAHE")
       
    def apply_bilateral_filter(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def apply_tophat(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (250, 250))
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    def apply_balckhat(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    def apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (15, 15), 0)
    
    def apply_intensity_normalization(self, image):
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def apply_preprocessing(self, color_space, method):
        return self.apply_operation(self.image, method, color_space)
    
    def apply_combinaison1(self, image):
        image_clahe = self.apply_clahe(image)
        return self.apply_bilateral_filter(image_clahe)
    
    def apply_combinaison2(self, image):
        filtered_image = self.apply_bilateral_filter(image)
        filtered_image_bgr = self.to_bgr(filtered_image)
        return self.apply_clahe(filtered_image_bgr)
        
    def apply_combinaison2(self, image):
        image_bilat = self.apply_bilateral_filter(image)
        return self.apply_clahe(image_bilat)
    
    def apply_combinaison3(self, image):
        image_clahe = self.apply_clahe(image)
        return self.apply_gaussian_blur(image_clahe)
    
    def apply_combinaison5(self, image):
        image_bilat = self.apply_bilateral_filter(image)
        return self.apply_tophat(image_bilat)

    def apply_sarki(self, image):
        image_enhancement = self.apply_sarki_preprocess(image, 10)
        return image_enhancement 
    
    def apply_sarki_preprocess(self, image, mu0):
        hsv_image = self.to_hsv()
        h, s, v = cv2.split(hsv_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        navg = np.mean(v_clahe)
        nCLIP = 3  
        nCL = nCLIP * navg #elle sert a quoi ?????
        mu0 = 128  # Intensité moyenne 
        muL = cv2.blur(v_clahe, (15, 15))  # Intensité locale moyenne
        # p' = p + mu0 - muL
        v_processed = np.clip(v_clahe + mu0 - muL, 0, 255)
        hsv_image_processed = cv2.merge([h, s, v_processed])
        image_bgr = hsv_image_processed.to_bgr()
        return image_bgr

    def fundus_autocrop(image: np.ndarray, mask=None):
        r_img = image[:, :, 0]
        threshold = 40
        _, roi = cv2.threshold(r_img, threshold, 1, cv2.THRESH_BINARY)
        not_null_pixels = cv2.findNonZero(roi)
        roi = roi.astype(np.uint8)
    
        if not_null_pixels is None:
            return {"image": image, "roi": roi}
    
        x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
        y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
        if mask is not None:
            if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
                return {"image": image, "roi": roi, "mask": mask}
        
            return {
                "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
                "roi": roi[y_range[0] : y_range[1], x_range[0] : x_range[1]],
                "mask": mask[y_range[0] : y_range[1], x_range[0] : x_range[1]]
            }
        else:
            if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
                return {"image": image, "roi": roi}
        
            return {
                "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
                "roi": roi[y_range[0] : y_range[1], x_range[0] : x_range[1]],
            }

    def apply_seoud(self, image):
        illumination = self.illumination_equalization(image, diameter=30)
        denoise = self.denoising(illumination)
        contrast = self.adaptive_contrast_equalization(denoise)
        return contrast 
    
    def mean_filter(self, image, kernel_size):
        filtered_image = cv2.blur(image, (kernel_size,kernel_size))
        return filtered_image
    
    def illumination_equalization(self, image, diameter):
        kernel_size = diameter
        #filtrage de chaque canal
        if image is None:
            print('Error loading the image')
        else:
            b, g, r = cv2.split(image)
            print(b.dtype)
            b_filtered = self.mean_filter(b,kernel_size)
            g_filtered = self.mean_filter(g,kernel_size)
            r_filtered = self.mean_filter(r,kernel_size)
        #moyenne de chaque canal
            mean_b = np.mean(b)
            mean_g = np.mean(g)
            mean_r = np.mean(r)
        #calcul
        #ajout moyenne
        #calcul final
            b_final = cv2.addWeighted(b,1,b_filtered,-1,mean_b)
            g_final = cv2.addWeighted(g,1,g_filtered,-1,mean_g)
            r_final = cv2.addWeighted(r,1,r_filtered,-1,mean_r)
        #fusion des images
            image_finale = cv2.merge([b_final, g_final, r_final])
            
            return image_finale
    
    def denoising(self, image): 
        b,g,r = cv2.split(image)
        kernel_size = 7
        b_filtered = self.mean_filter(b,kernel_size)
        g_filtered = self.mean_filter(g,kernel_size)
        r_filtered = self.mean_filter(r,kernel_size)
        denoising_image = cv2.merge([b_filtered, g_filtered, r_filtered])
        return denoising_image
    
    def adaptive_contrast_equalization(self, image): 
        b, g, r = cv2.split(image)
        E_x2_b = np.mean(b ** 2)
        E_x2_g = np.mean(g ** 2)
        E_x2_r = np.mean(r ** 2)
        E_x_b = np.mean(b)
        E_x_g = np.mean(g)
        E_x_r = np.mean(r)
        #Ecart-type
        std_b = E_x2_b - E_x_b
        std_g = E_x2_g - E_x_g
        std_r = E_x2_r - E_x_r
        #filtrage image avec 1 - h
        high_pass_filter = np.array([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]])/9
        convolution_b = cv2.filter2D(b, -1, 1-high_pass_filter)
        convolution_g = cv2.filter2D(g, -1, 1-high_pass_filter)
        convolution_r = cv2.filter2D(r, -1, 1-high_pass_filter)
        epsilon = 1e-7
        image_contrast_b = b + 1/(epsilon +std_b) * convolution_b
        image_contrast_g = g + 1/(epsilon +std_g) * convolution_g
        image_contrast_r = r + 1/(epsilon +std_r)* convolution_r
        image_finale = cv2.merge([image_contrast_b, image_contrast_g, image_contrast_r])
        return image_finale 
    




