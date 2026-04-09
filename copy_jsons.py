import os
import shutil
from pathlib import Path

# Nastavení cest (uprav podle potřeby)
JPG_DIR = Path("/Users/ondrejlukasek/Downloads/muni")
JSON_SOURCE_DIR = Path("/Users/ondrejlukasek/Documents/GitHub/KNN_project/out")
DEST_DIR = Path("/Users/ondrejlukasek/Documents/GitHub/KNN_project/out100muni")

def sync_json_files():
    # Vytvoření cílové složky, pokud neexistuje
    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True)
        print(f"Vytvořena složka: {DEST_DIR}")

    count = 0
    missing = 0

    # Procházíme soubory v první složce
    for jpg_file in JPG_DIR.glob("*_vis_error.jpg"):
        # Získáme základní ID (odstraníme '_vis_error.jpg')
        base_name = jpg_file.name.replace("_vis_error.jpg", "")
        json_filename = f"{base_name}.json"
        
        source_json = JSON_SOURCE_DIR / json_filename
        target_json = DEST_DIR / json_filename

        # Pokud JSON existuje, zkopírujeme ho
        if source_json.exists():
            shutil.copy2(source_json, target_json)
            count += 1
        else:
            print(f"Chybí JSON pro: {jpg_file.name}")
            missing += 1

    print("-" * 30)
    print(f"Hotovo! Zkopírováno: {count} JSON souborů.")
    if missing > 0:
        print(f"Nenalezeno: {missing} souborů.")

if __name__ == "__main__":
    sync_json_files()