# transformazioni/geometric.py
import cv2
import numpy as np

def skew_image(image, skew_factor=0.3):
    """
    Applica uno skew orizzontale all'intera immagine.
    skew_factor positivo: skew in un senso, negativo nell'altro.
    """
    h, w = image.shape[:2]
    
    # Matrice affine: spostiamo i vertici in orizzontale
    src_pts = np.float32([[0, 0], [w, 0], [0, h]])
    dst_pts = np.float32([
        [0, 0],
        [w + skew_factor * h, 0],
        [skew_factor * h, h]
    ])
    
    M = cv2.getAffineTransform(src_pts, dst_pts)
    skewed = cv2.warpAffine(image, M, (int(w + abs(skew_factor * h)), h))
    return skewed


def stretch_image(image, x_stretch=1.2, y_stretch=1.0):
    """
    Esegue uno stretching orizzontale/verticale (scalatura).
    """
    h, w = image.shape[:2]
    new_w = int(w * x_stretch)
    new_h = int(h * y_stretch)
    stretched = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return stretched


def perspective_transform(image):
    """
    Esempio di prospettiva “leggera” spostando leggermente i vertici.
    """
    h, w = image.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([
        [10, 10],
        [w - 10, 0],
        [0, h],
        [w, h - 10]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


def wave_effect(image, amplitude=5, frequency=20):
    """
    Esempio di effetto 'onda' orizzontale.
    amplitude: ampiezza dell'onda
    frequency: frequenza dell'onda
    """
    h, w = image.shape[:2]
    # Creiamo una mappa di coordinate spostate sinusoidalmente
    x_indices = np.arange(w)
    y_indices = np.arange(h)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    
    # Shift sinusoidale orizzontale
    # Più grande frequency → più onde
    shift_x = (amplitude * np.sin(2 * np.pi * y_grid / frequency)).astype(np.float32)
    map_x = (x_grid + shift_x).astype(np.float32)
    map_y = y_grid.astype(np.float32)
    
    # Remap
    waved = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return waved


def elastic_distortion(image, alpha=34, sigma=4):
    """
    Applicazione di una elastic distortion (semplificata).
    alpha e sigma controllano l'intensità della deformazione.
    """
    h, w = image.shape[:2]
    
    # Creiamo due mappe random di spostamento (dx, dy)
    random_state = np.random.RandomState(None)
    dx = cv2.GaussianBlur((random_state.rand(h, w) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(h, w) * 2 - 1), (17, 17), sigma) * alpha
    
    # Creiamo la mappa di lookup
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return distorted
