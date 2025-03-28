# transformazioni/effects.py
import cv2
import numpy as np

def gaussian_blur(image, ksize=5):
    """
    Applica un Gaussian Blur classico.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def motion_blur(image, kernel_size=15):
    """
    Crea un kernel lineare orizzontale e lo usa per sfocatura di movimento.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def ghosting_effect(image, alpha=0.5, offset=5):
    """
    Duplica l'immagine leggermente traslata e la sovrappone con trasparenza 'alpha'.
    offset: pixel di traslazione orizzontale/verticale (qui facciamo orizzontale).
    """
    h, w = image.shape[:2]
    shifted = np.zeros_like(image)
    
    if offset < w:
        shifted[:, offset:] = image[:, :w-offset]
    
    out = cv2.addWeighted(image, 1.0, shifted, alpha, 0)
    return out

def artificial_shadows(image, dark_factor=0.5):
    """
    Aggiunge un'ombra rettangolare casuale.
    """
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    
    x1 = np.random.randint(0, w)
    x2 = np.random.randint(x1, w)
    y1 = np.random.randint(0, h)
    y2 = np.random.randint(y1, h)
    
    # Applichiamo un fattore scuro solo nella zona [y1:y2, x1:x2]
    out[y1:y2, x1:x2] *= dark_factor
    
    return np.clip(out, 0, 255).astype(np.uint8)

def glitch_effect(image, shift_range=5):
    """
    Esempio di "glitch" duplicando i canali e spostandoli.
    """
    b, g, r = cv2.split(image)
    
    # Shift canale B
    rows, cols = b.shape[:2]
    M = np.float32([[1, 0, np.random.randint(-shift_range, shift_range)],
                    [0, 1, np.random.randint(-shift_range, shift_range)]])
    b_shifted = cv2.warpAffine(b, M, (cols, rows))
    
    # Shift canale R
    M2 = np.float32([[1, 0, np.random.randint(-shift_range, shift_range)],
                     [0, 1, np.random.randint(-shift_range, shift_range)]])
    r_shifted = cv2.warpAffine(r, M2, (cols, rows))
    
    # Ricombiniamo
    glitch_img = cv2.merge([b_shifted, g, r_shifted])
    return glitch_img
