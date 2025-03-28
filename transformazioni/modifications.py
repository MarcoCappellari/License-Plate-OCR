# transformazioni/modifications.py
import cv2
import numpy as np

def irregular_spacing(image, max_shift=2):
    """
    Simula uno spacing irregolare orizzontale lungo l'immagine
    (in pratica taglia verticalmente in piccole strisce e le sposta orizzontalmente).
    """
    h, w = image.shape[:2]
    out = np.zeros_like(image)
    num_slices = 10
    slice_height = h // num_slices
    
    for i in range(num_slices):
        y1 = i * slice_height
        y2 = h if i == num_slices-1 else (i+1) * slice_height
        
        shift = np.random.randint(-max_shift, max_shift+1)
        
        slice_img = image[y1:y2, :]
        
        # Calcoliamo i nuovi limiti
        x_start = max(0, shift)
        x_end = min(w, w + shift)  # shift potrebbe essere negativo
        
        # Copiamo i pixel con offset orizzontale
        out[y1:y2, x_start:x_end] = slice_img[:, : (x_end - x_start)]
    
    return out


def character_overlap(image, overlap_prob=0.1):
    """
    Simulazione grossolana di overlapping orizzontale di alcune bande verticali.
    Sovrappone un pezzo dell'immagine su un'altra zona.
    """
    out = image.copy()
    h, w = out.shape[:2]
    
    if np.random.rand() < overlap_prob:
        # 1) Scegliamo la larghezza della striscia (5..15), ma senza superare la larghezza totale
        max_strip = min(15, w)  # se w < 15, evitiamo di sforare
        if max_strip > 5:
            strip_width = np.random.randint(5, max_strip)  # 5..(max_strip-1)
            
            # 2) Selezioniamo x1 in modo che x1+strip_width <= w
            x1 = np.random.randint(0, w - strip_width + 1)
            region = out[:, x1:x1+strip_width]

            # 3) Selezioniamo x2 similmente, per incollare la striscia nello stesso spazio
            x2 = np.random.randint(0, w - strip_width + 1)
            out[:, x2:x2+strip_width] = region

    return out


def small_decorations(image, color=(0, 0, 255), thickness=1, num_lines=3):
    """
    Aggiunge delle piccole linee o puntini casuali sull'immagine
    (simulate "decorazioni" o "sporcature").
    """
    out = image.copy()
    h, w = out.shape[:2]
    
    for _ in range(num_lines):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = np.random.randint(0, w)
        y2 = np.random.randint(0, h)
        cv2.line(out, (x1,y1), (x2,y2), color, thickness)
    return out


def shape_modification(image, intensity=0.5):
    """
    Semplice esempio: applichiamo un morphing globale che 'allarga' o 'assottiglia' 
    i contorni, simulando leggere deformazioni dei caratteri.
    """
    # Convertiamo in grayscale e binarizziamo
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Kernel di morfologia
    kernel_size = int(3 + 5*intensity)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Se intensity > 0.5 facciamo dilation, altrimenti erosion
    if intensity >= 0.5:
        morph = cv2.dilate(binary, kernel, iterations=1)
    else:
        morph = cv2.erode(binary, kernel, iterations=1)
    
    # Riconvertiamo in immagine BGR mischiando col layer originale
    morph_bgr = cv2.cvtColor(255 - morph, cv2.COLOR_GRAY2BGR)
    # Combiniamo: dove c'è “nero” nel morph, lasciamo l'immagine originale, 
    # dove c'è “bianco” (caratteri) usiamo la parte "invertita"
    
    out = cv2.addWeighted(image, 0.5, morph_bgr, 0.5, 0)
    return out
