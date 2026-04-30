import json
import os
from openai import OpenAI

client = OpenAI()
BATCH_ID = "batch_69f34335d1588190aa7ebf6250d390ac" 
OUTPUT_DIR = "out_openai"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_batch_results(batch_id):
    print(f"Získávám informace o batche {batch_id}...")
    
    # 1. Zjistíme informace o dokončeném jobu
    job = client.batches.retrieve(batch_id)
    
    if job.status != "completed":
        print(f"Chyba: Batch je ve stavu '{job.status}', ne 'completed'.")
        return

    # 2. Stáhneme obsah výstupního souboru
    print("Stahuji data z OpenAI...")
    file_response = client.files.content(job.output_file_id)
    
    # 3. Rozstříháme JSONL na jednotlivé JSONy
    lines = file_response.text.strip().split('\n')
    print(f"Zpracovávám {len(lines)} výsledků...")

    for line in lines:
        data = json.loads(line)
        # custom_id je název původního souboru, který jsme tam poslali
        original_filename = data['custom_id']
        # Samotný obsah, který vygeneroval model
        content_string = data['response']['body']['choices'][0]['message']['content']
        
        # Převedeme string na skutečný JSON objekt pro hezké uložení
        json_content = json.loads(content_string)
        
        # Vytvoříme název souboru (např. strana_01.json)
        json_name = os.path.splitext(original_filename)[0] + ".json"
        output_path = os.path.join(OUTPUT_DIR, json_name)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=4)

    print(f"\nHotovo! Všechny soubory najdeš ve složce: {OUTPUT_DIR}")

if __name__ == "__main__":
    download_batch_results(BATCH_ID)
