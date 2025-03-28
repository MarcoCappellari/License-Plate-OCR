# transformazioni/color.py
import cv2
import numpy as np

def invert_colors(image):
    """
    Inverte i colori (effetto negativo).
    """
    return 255 - image

def reduce_contrast(image, factor=0.5):
    """
    Riduce il contrasto moltiplicando i valori per 'factor'.
    factor < 1: riduce contrasto
    factor > 1: aumenta contrasto
    """
    # convertScaleAbs = alpha * pixel + beta
    # Qui usiamo alpha=factor, beta=0
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def extreme_saturation(image, saturation_scale=2.0):
    """
    Aumenta o diminuisce drasticamente la saturazione (solo su immagini in HSV).
    saturation_scale > 1 = saturazione più intensa
    saturation_scale < 1 = saturazione minore
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Convertiamo in float per evitare overflow
    s = s.astype(np.float32)
    s *= saturation_scale
    s = np.clip(s, 0, 255)
    
    new_hsv = cv2.merge([h, s.astype(np.uint8), v])
    out = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
    return out

def custom_rgb_filter(image, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    """
    Applica un fattore di scala ai canali R, G, B.
    """
    b, g, r = cv2.split(image)
    b = cv2.convertScaleAbs(b, alpha=b_factor, beta=0)
    g = cv2.convertScaleAbs(g, alpha=g_factor, beta=0)
    r = cv2.convertScaleAbs(r, alpha=r_factor, beta=0)
    return cv2.merge([b, g, r])

def gradient_overlay(image):
    """
    Aggiunge una leggera sfumatura (gradient) sull'immagine.
    """
    h, w = image.shape[:2]
    # Creiamo un gradiente orizzontale dal nero al bianco
    gradient = np.linspace(0, 1, w, dtype=np.float32)
    gradient = np.tile(gradient, (h, 1))
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertiamo in float e normalizziamo
        img_float = image.astype(np.float32) / 255.0
        
        # Sovrapponiamo gradiente su uno dei canali (ad es. canale B)
        img_float[:,:,0] = img_float[:,:,0] * gradient
        
        # Convertiamo di nuovo in 0-255
        result = (img_float * 255).astype(np.uint8)
        return result
    else:
        # Caso immagine grayscale
        img_float = image.astype(np.float32) / 255.0
        img_float = img_float * gradient
        return (img_float * 255).astype(np.uint8)
