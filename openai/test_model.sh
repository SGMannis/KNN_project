#!/bin/bash

# --- NASTAVENÍ ---
PY_SCRIPT="gpt_inference.py" 
INPUT_FILE="test_list.txt"

# Cesty
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PY_PATH="$SCRIPT_DIR/$PY_SCRIPT"

echo "=== 1. KROK: SPUŠTĚNÍ OCR (LOOP) ==="

# Kontrola souborů
if [ ! -f "$PY_PATH" ]; then
    echo "CHYBA: $PY_PATH nenalezen!"
    exit 1
fi

# Smyčka pro OCR
while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" ]] && continue
    
    echo ">>> Analyzuji: $line"
    python "$PY_PATH" -i "$line"
    
done < "$INPUT_FILE"

echo -e "\n=== 2. KROK: VIZUALIZACE ==="

# Přesuneme se o složku výš (do rootu projektu)
cd "$SCRIPT_DIR/.."
echo "Aktuální složka: $(pwd)"

# Spuštění vizualizace na celou složku najednou
python visualize_matches.py -j openai/out -o openai/out_vis

echo "=== VŠE HOTOVO ==="
