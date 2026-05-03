import json
import os
import argparse
from pathlib import Path

def transform_json(data):
    """
    Pouze vytáhne seznam ze slovníku pod klíčem 'chapters'.
    Pokud už je data seznam, vrátí ho beze změny.
    """
    if isinstance(data, dict) and "chapters" in data:
        return data["chapters"]
    return data

def process_directory(directory_path):
    path = Path(directory_path)
    
    if not path.is_dir():
        print(f"Chyba: {directory_path} není platná složka.")
        return

    json_files = list(path.glob("*.json"))
    
    if not json_files:
        print("Ve složce nebyly nalezeny žádné .json soubory.")
        return

    print(f"Zpracovávám {len(json_files)} souborů...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            # Provedeme pouze vybalení seznamu z klíče "chapters"
            transformed_content = transform_json(content)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_content, f, ensure_ascii=False, indent=4)
            
            print(f"OK: {file_path.name}")
        
        except Exception as e:
            print(f"CHYBA u souboru {file_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Převede formát JSONů z objektu {'chapters': [...]} na prostý seznam.")
    parser.add_argument("folder", help="Cesta ke složce s JSON soubory")
    
    args = parser.parse_args()
    process_directory(args.folder)
