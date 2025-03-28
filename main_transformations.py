"""
main_transformations.py
Esegue:
1) Lettura ground truth (eu, eu2, eu3).
2) Applicazione di trasformazioni su ciascuna immagine (ognuna con vari "livelli" di aggressività).
3) Salvataggio delle immagini modificate in sottocartelle.
4) OCR sulle immagini modificate.
5) Confronto e statistiche finali.
"""

import os
import sys
import cv2
import json
import re
from difflib import SequenceMatcher

# 1) Disabilita i log di onnxruntime e open_image_models
import warnings
warnings.filterwarnings("ignore")  # Ignora eventuali Python warnings

import logging
# Imposta i messaggi di open_image_models (o l'intero "open_image_models") a livello ERROR
logging.getLogger("open_image_models").setLevel(logging.ERROR)
logging.getLogger("open_image_models.detection.core.yolo_v9.inference").disabled = True

# 2) Imposta su CPU e alza il livello di severità di ONNXRuntime
os.environ["ORT_DISABLE_ALL_EXCEPT_CPU"] = "1"
import onnxruntime as ort
ort.set_default_logger_severity(4)  # 4 = Fatal, nasconde Warning/Error di livello minore

from fast_alpr import ALPR

# Import delle trasformazioni
from transformazioni.geometric import (
    skew_image, wave_effect, elastic_distortion, stretch_image, perspective_transform
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
    Legge la ground truth da un file .txt in formato ... <plate>.
    Ritorna la targa (maiuscola).
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
    Legge un file JSON di label, ad es. in eu2/ o eu3/.
    Ritorna un dict { "filename.jpg": "PLATE" }
    """
    gt_dict = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                image = entry["image"]
                # Esempio: "48AGR391_FRONT.jpg" => targa "48AGR391"
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
      "eu":  { "1.jpg": "ABC123", "2.jpg": "..." },
      "eu2": { "48AGR391_FRONT.jpg": "48AGR391", ... },
      "eu3": ...
    }
    """
    all_folders = {
        "eu": {},
        "eu2": parse_ground_truth_json("eu2/labels.json"),
        "eu3": parse_ground_truth_json("eu3/labels.json"),
        # Aggiungi qui se hai altre cartelle (eu4, eu5, ecc.)
    }

    # Per 'eu', leggiamo i .txt (se la cartella esiste)
    if os.path.isdir("eu"):
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
    Ritorna (matched_chars, total_chars)
    per calcolare l'accuracy carattere per carattere.
    """
    sm = SequenceMatcher(None, predicted, actual)
    matched = sum(block.size for block in sm.get_matching_blocks())
    total = max(len(predicted), len(actual))
    return matched, total

def run_ocr(alpr, image_path):
    """
    Esegue OCR con fast_alpr e ritorna il testo riconosciuto (in maiuscolo).
    """
    try:
        results = alpr.predict(image_path)
        ocr_plate = results[0].ocr.text.strip().upper() if results else ""
    except Exception as e:
        print(f"Errore OCR con {image_path}: {e}")
        ocr_plate = ""
    return ocr_plate

########################################################################
# --- TRASFORMAZIONI CON LIVELLI DI AGGRESSIVITA' --- #
########################################################################

"""
Di seguito un elenco di trasformazioni, ognuna con:
 - "name"        : nome della trasformazione (es. "wave")
 - "levels"      : lista di aggressività in [0..1] (es. [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
 - "apply_func"  : funzione(img, level) -> immagine trasformata
   (decidi tu come interpretare 'level' nella trasformazione)
"""

########################################################################
# --- TRASFORMAZIONI CON LIVELLI DI AGGRESSIVITA' --- #
########################################################################

TRANSFORMATIONS_WITH_LEVELS = [
    {
        "name": "skew",
        "levels": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        # Esempio: fattore skew = lvl
        "apply_func": lambda img, lvl: skew_image(img, skew_factor=lvl)
    },
    {
        "name": "stretch",
        "levels": [0.1, 0.3, 0.5, 0.7, 1.0],
        # x_stretch = 1 + lvl
        "apply_func": lambda img, lvl: stretch_image(img, x_stretch=(1.0 + lvl), y_stretch=1.0)
    },
    {
        "name": "perspective",
        "levels": [0.2, 0.5, 1.0],
        # Non c'è un param di perspective transform. Creiamo "fake" intensità e la applichiamo in base a lvl
        "apply_func": lambda img, lvl: perspective_transform(img)  # ignora lvl o usalo in un perspective custom
    },
    {
        "name": "wave",
        "levels": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "apply_func": lambda img, lvl: wave_effect(img, amplitude=int(10 * lvl), frequency=40)
    },
    {
        "name": "elastic",
        "levels": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "apply_func": lambda img, lvl: elastic_distortion(img, alpha=int(34 * lvl), sigma=6)
    },
    {
        "name": "invert",
        "levels": [1.0],  # un solo "livello" (non param.)
        "apply_func": lambda img, lvl: invert_colors(img)
    },
    {
        "name": "reduce_contrast",
        "levels": [0.3, 0.5, 0.7, 1.0, 1.2],
        # factor < 1 = riduce, >1 = aumenta
        "apply_func": lambda img, lvl: reduce_contrast(img, factor=lvl)
    },
    {
        "name": "extreme_saturation",
        "levels": [0.5, 1.0, 1.5, 2.0, 2.5],
        # interpretare lvl come "saturation_scale"
        "apply_func": lambda img, lvl: extreme_saturation(img, saturation_scale=lvl)
    },
    {
        "name": "custom_rgb",
        "levels": [0.5, 1.0, 1.5],
        # ad es. r_factor = 1 + lvl, g_factor=1-lvl, ...
        "apply_func": lambda img, lvl: custom_rgb_filter(img, r_factor=1+lvl, g_factor=1-lvl, b_factor=1.0)
    },
    {
        "name": "gradient_overlay",
        "levels": [1.0],  # singolo livello (non param.)
        "apply_func": lambda img, lvl: gradient_overlay(img)
    },
    {
        "name": "gaussian_noise",
        "levels": [0.1, 0.3, 0.5, 0.7, 1.0],
        # var = 20*(lvl)
        "apply_func": lambda img, lvl: gaussian_noise(img, mean=0, var=20*lvl)
    },
    {
        "name": "salt_and_pepper",
        "levels": [0.01, 0.03, 0.05, 0.07, 0.1],
        "apply_func": lambda img, lvl: salt_and_pepper(img, amount=lvl)
    },
    {
        "name": "perlin_noise",
        "levels": [0.2, 0.4, 0.6, 0.8, 1.0],
        # scale = 50*lvl
        "apply_func": lambda img, lvl: perlin_noise(img, scale=int(50*lvl))
    },
    {
        "name": "jpeg_artifacts",
        "levels": [30, 50, 70, 90],  # interpretati come quality
        "apply_func": lambda img, lvl: jpeg_artifacts(img, quality=int(lvl))
    },
    {
        "name": "irregular_spacing",
        "levels": [1.0],  # un solo livello
        "apply_func": lambda img, lvl: irregular_spacing(img, max_shift=3)
    },
    {
        "name": "char_overlap",
        "levels": [1.0],
        "apply_func": lambda img, lvl: character_overlap(img, overlap_prob=0.8)
    },
    {
        "name": "small_decorations",
        "levels": [1.0],
        "apply_func": lambda img, lvl: small_decorations(img, color=(0,0,255), thickness=1, num_lines=3)
    },
    {
        "name": "shape_modif",
        "levels": [0.3, 0.5, 0.7, 1.0],
        # intensità => 0.3..1.0
        "apply_func": lambda img, lvl: shape_modification(img, intensity=lvl)
    },
    {
        "name": "gaussian_blur",
        "levels": [1.0],  # un solo livello
        "apply_func": lambda img, lvl: gaussian_blur(img, ksize=5)
    },
    {
        "name": "motion_blur",
        "levels": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "apply_func": lambda img, lvl: motion_blur(img, kernel_size=int(3 + 12*lvl))
    }
]


########################################################################
# --- FUNZIONE PRINCIPALE --- #
########################################################################

def main():
    # 1) Carichiamo le mappe di ground truth
    all_gt = load_ground_truths()

    # 2) Inizializziamo l'OCR
    alpr = ALPR()

    # Variabili globali per statistiche totali (tutte le cartelle, trasformazioni e livelli)
    global_total = 0
    global_correct = 0
    global_matched_chars = 0
    global_total_chars = 0

    # 3) Iteriamo su ogni cartella (eu, eu2, eu3, ...)
    for folder_name, gt_map in all_gt.items():
        if not os.path.isdir(folder_name):
            continue

        # Lista di immagini .jpg
        image_files = [f for f in os.listdir(folder_name) if f.lower().endswith(".jpg")]
        image_files.sort()

        print(f"\n=== PROCESSING FOLDER: {folder_name} ===")

        # 4) Per ogni trasformazione con livelli
        for transf in TRANSFORMATIONS_WITH_LEVELS:
            transf_name = transf["name"]
            levels = transf["levels"]
            apply_func = transf["apply_func"]

            # Creiamo una cartella di output (es: "output_wave") dentro la cartella "folder_name"
            transf_output_dir = os.path.join(folder_name, f"output_{transf_name}")
            os.makedirs(transf_output_dir, exist_ok=True)

            # Statistiche su TUTTI i livelli di questa trasformazione
            transf_tot = 0
            transf_correct = 0
            transf_matched = 0
            transf_chars = 0

            # 5) Cicliamo i livelli di aggressività
            for lvl in levels:
                # Es. lvl=0.3 => perc_int=30 => "level_30"
                perc_int = int(lvl * 100)
                level_folder_name = f"level_{perc_int}"
                level_output_dir = os.path.join(transf_output_dir, level_folder_name)
                os.makedirs(level_output_dir, exist_ok=True)

                # Statistiche per questo specifico livello
                level_tot = 0
                level_correct = 0
                level_matched = 0
                level_chars = 0

                # 6) Elaboriamo ogni immagine
                for img_file in image_files:
                    true_plate = gt_map.get(img_file)
                    if not true_plate:
                        continue

                    image_path = os.path.join(folder_name, img_file)
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    # Applica la trasformazione con intensità lvl
                    transformed_img = apply_func(img, lvl)

                    # Salva l'immagine modificata
                    # Nome file: "<original>_<transfName>_<percInt>.jpg"
                    out_filename = f"{os.path.splitext(img_file)[0]}_{transf_name}_{perc_int}.jpg"
                    out_path = os.path.join(level_output_dir, out_filename)
                    cv2.imwrite(out_path, transformed_img)

                    # OCR
                    ocr_plate = run_ocr(alpr, out_path)

                    # Confronto
                    level_tot += 1
                    if ocr_plate == true_plate:
                        level_correct += 1

                    matched, tot_chars = sequence_match_score(ocr_plate, true_plate)
                    level_matched += matched
                    level_chars += tot_chars

                # Fine loop immagini per un livello
                transf_tot += level_tot
                transf_correct += level_correct
                transf_matched += level_matched
                transf_chars += level_chars

                # Aggiorniamo le globali
                global_total += level_tot
                global_correct += level_correct
                global_matched_chars += level_matched
                global_total_chars += level_chars

                # Stampiamo i risultati di questo livello
                if level_tot > 0:
                    pct_correct = (level_correct / level_tot) * 100
                    pct_chars = (level_matched / level_chars) * 100
                    print(f"  [{transf_name}] lvl={perc_int}% | "
                          f"Imgs: {level_tot} | Perfette: {level_correct} ({pct_correct:.2f}%) "
                          f"| Chars: {pct_chars:.2f}%")
                else:
                    print(f"  [{transf_name}] lvl={perc_int}% | Nessuna immagine elaborata.")

            # Fine loop su tutti i "levels" di questa trasformazione

            # Statistiche totali per questa trasformazione (somma di tutti i livelli)
            if transf_tot > 0:
                total_pct_correct = (transf_correct / transf_tot) * 100
                total_pct_chars = (transf_matched / transf_chars) * 100
                print(f" => [{transf_name}] TOT (tutti i livelli) | "
                      f"Imgs: {transf_tot} | Perfette: {transf_correct} ({total_pct_correct:.2f}%) "
                      f"| Chars: {total_pct_chars:.2f}%")
            else:
                print(f" => [{transf_name}] Nessuna immagine processata in totale.")

    # 7) Statistiche globali finali su tutte le cartelle e trasformazioni
    print("\n=== RISULTATI FINALI SU TUTTE LE CARTELLE E TRASFORMAZIONI ===")
    if global_total > 0:
        global_perc_correct = (global_correct / global_total) * 100
        global_char_acc = (global_matched_chars / global_total_chars) * 100
        print(f"Immagini totali elaborate: {global_total}")
        print(f"Targhe perfettamente corrette: {global_correct} ({global_perc_correct:.2f}%)")
        print(f"Accuratezza carattere per carattere: {global_char_acc:.2f}%")
    else:
        print("Nessuna immagine processata.")

if __name__ == "__main__":
    main()
