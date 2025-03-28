# transformazioni/noise.py
import cv2
import numpy as np

def gaussian_noise(image, mean=0, var=10):
    """
    Aggiunge rumore gaussiano all'immagine.
    mean: media del rumore
    var: varianza del rumore
    """
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    
    # Se l'immagine è BGR, gauss avrà 3 canali
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def salt_and_pepper(image, amount=0.02):
    """
    Aggiunge rumore 'salt and pepper' all'immagine.
    amount: percentuale di pixel corrotti.
    """
    out = image.copy()
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))
    
    # Aggiunta sale (pixel bianchi)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    out[coords[0], coords[1]] = (255,255,255) if len(image.shape) == 3 else 255
    
    # Aggiunta pepe (pixel neri)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    out[coords[0], coords[1]] = (0,0,0) if len(image.shape) == 3 else 0
    return out

def perlin_noise(image, scale=50):
    """
    Esempio base di Perlin noise (semplificato).
    Per un controllo più fine occorrerebbe una libreria dedicata (perlin-noise, noise, ecc.).
    """
    h, w = image.shape[:2]
    
    # Generiamo una "texture" di perlin noise semplificato
    # Qui usiamo un random come placeholder
    # -> Per un vero Perlin bisognerebbe usare funzioni che generano gradiente liscio.
    noise = np.random.uniform(0,1,(h, w)).astype(np.float32)
    
    # Espandiamo a 3 canali se necessario
    if len(image.shape) == 3 and image.shape[2] == 3:
        noise_3c = np.dstack([noise, noise, noise])
    else:
        noise_3c = noise
    
    # Combiniamo il noise con l'immagine
    img_float = image.astype(np.float32)
    combined = img_float + noise_3c * scale
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    return combined

def jpeg_artifacts(image, quality=30):
    """
    Simula artefatti di compressione JPEG ricomprimendo l'immagine con bassa qualità.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)  # 1 = BGR
    return decimg
