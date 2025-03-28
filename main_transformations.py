"""
main_transformations.py
Esegue:
1) Lettura ground truth (eu, eu2, eu3).
2) Applicazione di trasformazioni su ciascuna immagine (creazione di 20 cartelle).
3) OCR sulle immagini modificate.
4) Confronto e statistiche.
"""

import os
import sys
import cv2
import json
import re
from difflib import SequenceMatcher
from collections import defaultdict

# 1) Disabilita i log di onnxruntime e open_image_models
import warnings
warnings.filterwarnings("ignore")  # Ignora eventuali Python warnings

import logging
# Imposta i messaggi di open_image_models (o l'intero "open_image_models" se preferisci) a livello ERROR
logging.getLogger("open_image_models").setLevel(logging.ERROR)
logging.getLogger("open_image_models.detection.core.yolo_v9.inference").disabled = True


# 2) Imposta su CPU e alza il livello di severità di ONNXRuntime
os.environ["ORT_DISABLE_ALL_EXCEPT_CPU"] = "1"
import onnxruntime as ort
ort.set_default_logger_severity(4)  # 4 = Fatal, quindi nasconde Warning/Error di livello minore

from fast_alpr import ALPR

# Importiamo le trasformazioni
from transformazioni.geometric import (
    skew_image, stretch_image, perspective_transform, wave_effect, elastic_distortion
)
from transformazioni.color import (
    invert_colors, reduce_contrast, extreme_saturation, custom_rgb_filter, gradient_overlay
)
from transformazioni.noise import (
    gaussian_noise, salt_and_pepper, perlin_noise, jpeg_artifacts
)
from transformazioni.modifications import (
    irregular_spacing, character_overlap, small_decorations, shape_modification
)
from transformazioni.effects import (
    gaussian_blur, motion_blur, ghosting_effect, artificial_shadows, glitch_effect
)


########################################################################
# --- FUNZIONI DI SUPPORTO PER LA GROUND TRUTH --- #
########################################################################

def parse_ground_truth_txt(txt_path):
    """
    Legge la ground truth da un file .txt in formato:  ...    <plate>
    Ritorna la targa come stringa
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                return parts[5].strip().upper()
    except Exception as e:
        print(f"Errore nella lettura di {txt_path}: {e}")
    return None

def parse_ground_truth_json(json_path):
    """
    Legge un file JSON di label, ad es. in eu2/ e eu3/.
    Ritorna un dict { "filename.jpg": "PLATE" }
    """
    gt_dict = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                image = entry["image"]
                match = re.search(r'_([A-Z0-9]{5,})_FRONT', image)
                if match:
                    plate = match.group(1).upper()
                    gt_dict[image] = plate
    except Exception as e:
        print(f"Errore nella lettura del JSON {json_path}: {e}")
    return gt_dict

def load_ground_truths():
    """
    Carica le mappe di ground truth per 'eu', 'eu2', 'eu3', etc.
    Ritorna un dizionario del tipo:
    {
       "eu": { "1.jpg": "ABC123", "2.jpg": "..." },
       "eu2": { "48AGR391_FRONT.jpg": "48AGR391", ... },
       "eu3": ...
    }
    """
    all_folders = {
        "eu": {},
        "eu2": parse_ground_truth_json("eu2/labels.json"),
        "eu3": parse_ground_truth_json("eu3/labels.json"),
        # se vuoi anche eu4, eu5:
        # "eu4": parse_ground_truth_json("eu4/labels.json"),
        # "eu5": parse_ground_truth_json("eu5/labels.json"),
    }

    # Per 'eu', leggiamo i .txt
    for f in os.listdir("eu"):
        if f.lower().endswith(".txt"):
            image_file = f.replace(".txt", ".jpg")
            full_path = os.path.join("eu", f)
            plate = parse_ground_truth_txt(full_path)
            if plate:
                all_folders["eu"][image_file] = plate

    return all_folders

########################################################################
# --- FUNZIONI PER L'ANALISI OCR E STATISTICHE --- #
########################################################################

def sequence_match_score(predicted, actual):
    """
    Ritorna numero di caratteri coincidenti e numero totale di caratteri (per calcolo accuracy).
    """
    sm = SequenceMatcher(None, predicted, actual)
    matched_chars = sum(block.size for block in sm.get_matching_blocks())
    total_chars = max(len(predicted), len(actual))
    return matched_chars, total_chars

def run_ocr(alpr, image_path):
    """
    Esegue OCR con fast_alpr e ritorna il testo riconosciuto (maiuscolo).
    """
    try:
        results = alpr.predict(image_path)
        ocr_plate = results[0].ocr.text.strip().upper() if results else ""
    except Exception as e:
        print(f"Errore OCR con {image_path}: {e}")
        ocr_plate = ""
    return ocr_plate

########################################################################
# --- LISTA DELLE TRASFORMAZIONI DA APPLICARE --- #
########################################################################

TRANSFORMATIONS = [
    ("skew", lambda img: skew_image(img, skew_factor=0.3)),
    ("stretch", lambda img: stretch_image(img, x_stretch=1.2, y_stretch=1.0)),
    ("perspective", lambda img: perspective_transform(img)),
    ("wave", lambda img: wave_effect(img, amplitude=10, frequency=40)),
    ("elastic", lambda img: elastic_distortion(img, alpha=34, sigma=6)),
    ("invert", lambda img: invert_colors(img)),
    ("reduce_contrast", lambda img: reduce_contrast(img, factor=0.5)),
    ("extreme_saturation", lambda img: extreme_saturation(img, saturation_scale=2.0)),
    ("custom_rgb", lambda img: custom_rgb_filter(img, r_factor=1.5, g_factor=0.8, b_factor=1.0)),
    ("gradient_overlay", lambda img: gradient_overlay(img)),
    ("gaussian_noise", lambda img: gaussian_noise(img, mean=0, var=20)),
    ("salt_and_pepper", lambda img: salt_and_pepper(img, amount=0.02)),
    ("perlin_noise", lambda img: perlin_noise(img, scale=50)),
    ("jpeg_artifacts", lambda img: jpeg_artifacts(img, quality=30)),
    ("irregular_spacing", lambda img: irregular_spacing(img, max_shift=3)),
    ("char_overlap", lambda img: character_overlap(img, overlap_prob=0.8)),
    ("small_decorations", lambda img: small_decorations(img, color=(0, 0, 255), thickness=1, num_lines=3)),
    ("shape_modif", lambda img: shape_modification(img, intensity=0.6)),
    ("gaussian_blur", lambda img: gaussian_blur(img, ksize=5)),
    ("motion_blur", lambda img: motion_blur(img, kernel_size=15)),
    # Aggiungi altre se vuoi (ghosting_effect, artificial_shadows, glitch_effect, ecc.)
]

########################################################################
# --- FUNZIONE PRINCIPALE --- #
########################################################################

def main():
    # 1) Carichiamo le mappe di ground truth
    all_gt = load_ground_truths()

    # 2) Istanziamo l'OCR
    alpr = ALPR()

    # Variabili per statistiche globali
    global_total = 0
    global_correct = 0
    global_matched_chars = 0
    global_total_chars = 0

    # Per ogni cartella (eu, eu2, eu3, ecc.)
    for folder_name, gt_map in all_gt.items():
        # Otteniamo la lista di immagini .jpg
        image_files = [f for f in os.listdir(folder_name) if f.lower().endswith(".jpg")]
        image_files.sort()

        print(f"\n=== PROCESSING FOLDER: {folder_name} ===")

        # Per ogni trasformazione definita
        for transf_name, transf_func in TRANSFORMATIONS:
            # Creiamo la cartella di output (es: "output_{transf_name}")
            output_dir = os.path.join(folder_name, f"output_{transf_name}")
            os.makedirs(output_dir, exist_ok=True)

            # Statistiche per questa trasformazione
            t_tot = 0
            t_correct = 0
            t_matched = 0
            t_chars = 0

            for img_file in image_files:
                true_plate = gt_map.get(img_file)
                if not true_plate:
                    # Se non abbiamo ground truth per questa immagine, saltiamo
                    continue

                image_path = os.path.join(folder_name, img_file)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                # Applichiamo la trasformazione
                transformed_img = transf_func(img)

                # Salviamo l'immagine trasformata
                out_filename = f"{os.path.splitext(img_file)[0]}_{transf_name}.jpg"
                out_path = os.path.join(output_dir, out_filename)
                cv2.imwrite(out_path, transformed_img)

                # Ora eseguiamo OCR sull'immagine trasformata
                ocr_plate = run_ocr(alpr, out_path)

                # Confronto
                t_tot += 1
                if ocr_plate == true_plate:
                    t_correct += 1

                matched, total_chars = sequence_match_score(ocr_plate, true_plate)
                t_matched += matched
                t_chars += total_chars

            # Stampa statistiche per questa trasformazione
            if t_tot > 0:
                perc_correct = t_correct / t_tot * 100
                perc_char_acc = t_matched / t_chars * 100
                print(f"  >> {transf_name} | Immagini: {t_tot} | Perfette: {t_correct} ({perc_correct:.2f}%) | Chars: {perc_char_acc:.2f}%")
            else:
                print(f"  >> {transf_name} | Nessuna immagine processata.")

            # Aggiungiamo al globale
            global_total += t_tot
            global_correct += t_correct
            global_matched_chars += t_matched
            global_total_chars += t_chars

    # 3) Statistiche globali finali
    print("\n=== RISULTATI FINALI SU TUTTE LE CARTELLE E TRASFORMAZIONI ===")
    if global_total > 0:
        global_perc_correct = global_correct / global_total * 100
        global_char_acc = global_matched_chars / global_total_chars * 100
        print(f"Immagini totali elaborate: {global_total}")
        print(f"Targhe perfettamente corrette: {global_correct} ({global_perc_correct:.2f}%)")
        print(f"Accuratezza carattere per carattere: {global_char_acc:.2f}%")
    else:
        print("Nessuna immagine processata.")

if __name__ == "__main__":
    main()
