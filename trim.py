import os
import json
import argparse
import glob

def clean_data(json_dir, img_dir):
    # Kontrola existencí složek
    if not os.path.isdir(json_dir):
        print(f"Chyba: Složka s JSONy '{json_dir}' neexistuje.")
        return
    if not os.path.isdir(img_dir):
        print(f"Chyba: Složka s obrázky '{img_dir}' neexistuje.")
        return

    # Najdeme všechny JSON soubory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    total_jsons_start = len(json_files)
    deleted_count = 0

    for json_path in json_files:
        should_delete = False
        
        # 1. Kontrola, jestli je soubor úplně prázdný (0 bajtů)
        if os.path.getsize(json_path) == 0:
            should_delete = True
        else:
            # 2. Kontrola, jestli obsahuje jen []
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content == "[]":
                        should_delete = True
            except Exception as e:
                print(f"Varování: Nepodařilo se přečíst {json_path}: {e}")

        if should_delete:
            # Získáme název souboru bez přípony pro smazání obrázku
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            img_path = os.path.join(img_dir, base_name + ".jpg")

            # Nejdříve smažeme obrázek, pokud existuje
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Chyba při mazání obrázku {img_path}: {e}")
            
            # Poté smažeme JSON
            try:
                os.remove(json_path)
                deleted_count += 1
            except Exception as e:
                print(f"Chyba při mazání JSONu {json_path}: {e}")

    # Finální přepočítání souborů v obou složkách pro ověření parity
    final_jsons = len(glob.glob(os.path.join(json_dir, "*.json")))
    final_imgs = len(glob.glob(os.path.join(img_dir, "*.jpg")))

    # Výpis statistik
    print("="*40)
    print(f"ÚKLID DATOVÉ SADY")
    print("="*40)
    print(f"Smazáno párů (JSON + JPG):      {deleted_count}")
    print("-"*40)
    print(f"Zbývající počet JSONů:          {final_jsons}")
    print(f"Zbývající počet obrázků:        {final_imgs}")
    print("-"*40)

    if final_jsons == final_imgs:
        print("STAV: OK (Počty souborů souhlasí)")
    else:
        print("STAV: VAROVÁNÍ (Počty souborů se liší!)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smaže nevalidní JSONy a jejich odpovídající obrázky.")
    parser.add_argument("-j", "--json_dir", type=str, required=True, help="Cesta ke složce s JSON soubory")
    parser.add_argument("-i", "--img_dir", type=str, required=True, help="Cesta ke složce s JPG obrázky")
    
    args = parser.parse_args()
    clean_data(args.json_dir, args.img_dir)
    