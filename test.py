import os
import sys
import contextlib
import re
import json
from difflib import SequenceMatcher
from collections import defaultdict

os.environ["ORT_DISABLE_ALL_EXCEPT_CPU"] = "1"
import onnxruntime as ort
from fast_alpr import ALPR

ort.set_default_logger_severity(3)

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def parse_ground_truth_txt(txt_path):
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

def process_folder(folder_path, ground_truth_map):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    image_files = sorted(image_files, key=extract_number)

    totali = 0
    corretti = 0
    caratteri_corretti = 0
    caratteri_totali = 0

    with suppress_stderr():
        alpr = ALPR()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        true_plate = ground_truth_map.get(image_file)

        if not true_plate:
            print(f"{image_file} → ⚠️ Nessuna targa trovata nei dati di ground truth")
            continue

        with suppress_stderr():
            try:
                results = alpr.predict(image_path)
                ocr_plate = results[0].ocr.text.strip().upper() if results else ""
            except Exception as e:
                print(f"Errore con {image_file}: {e}")
                continue

        totali += 1
        status = "✅" if ocr_plate == true_plate else "❌"
        print(f"{status} {image_file} → OCR: {ocr_plate} | CORRETTA: {true_plate}")

        if ocr_plate == true_plate:
            corretti += 1

        sm = SequenceMatcher(None, ocr_plate, true_plate)
        caratteri_corretti += sum(block.size for block in sm.get_matching_blocks())
        caratteri_totali += max(len(ocr_plate), len(true_plate))

    return totali, corretti, caratteri_corretti, caratteri_totali

def main():
    all_folders = {
        "eu": {},  # verrà popolato da .txt
        "eu2": parse_ground_truth_json("eu2/labels.json"),
        "eu3": parse_ground_truth_json("eu3/labels.json"),
        #"eu4": parse_ground_truth_json("eu4/labels.json"),
        #"eu5": parse_ground_truth_json("eu5/labels.json"),
    }

    # Per 'eu' leggiamo i .txt
    for f in os.listdir("eu"):
        if f.lower().endswith(".txt"):
            image_file = f.replace(".txt", ".jpg")
            full_path = os.path.join("eu", f)
            plate = parse_ground_truth_txt(full_path)
            if plate:
                all_folders["eu"][image_file] = plate

    totali = corretti = caratteri_corretti = caratteri_totali = 0

    for folder, gt_map in all_folders.items():
        print(f"\n📂 Analisi cartella: {folder}")
        t, c, cc, ct = process_folder(folder, gt_map)
        totali += t
        corretti += c
        caratteri_corretti += cc
        caratteri_totali += ct

    print("\n📊 RISULTATI FINALI:")
    print(f"Totale immagini: {totali}")
    print(f"Targhe perfettamente corrette: {corretti} ({corretti / totali:.2%})")
    print(f"Accuratezza carattere per carattere: {caratteri_corretti / caratteri_totali:.2%}")

if __name__ == "__main__":
    main()
