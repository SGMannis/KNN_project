#!/bin/bash

# Název souboru, kde máš ty cesty k obrázkům
INPUT_FILE="test_list.txt"

# Kontrola, jestli seznam existuje
if [ ! -f "$INPUT_FILE" ]; then
    echo "Chyba: Soubor $INPUT_FILE nebyl nalezen!"
    exit 1
fi

echo "=== START AUTOMATIZACE ==="

# Čtení souboru řádek po řádku
while IFS= read -r line || [ -n "$line" ]; do
    # Přeskočí prázdné řádky
    [[ -z "$line" ]] && continue
    
    echo "Právě zpracovávám: $line"
    
    # Spuštění tvého python skriptu
    # -i je cesta k obrázku z řádku v textovém souboru
    python gpt_inference.py -i "$line"
    
    echo "--------------------------"
done < "$INPUT_FILE"

echo "=== VŠECHNO HOTOVO ==="
