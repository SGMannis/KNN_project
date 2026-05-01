import json
import os
from openai import OpenAI

# --- KONFIGURACE ---
client = OpenAI()
BATCH_LIST_FILE = "4o.txt"
OUTPUT_DIR = "out_4o"
# -------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_batch_results(batch_id):
    print(f"\n--- Zpracovávám batch {batch_id} ---")
    
    try:
        # 1. Zjistíme informace o jobu
        job = client.batches.retrieve(batch_id)
        
        if job.status != "completed":
            print(f"  [!] Přeskakuji: Batch je ve stavu '{job.status}'.")
            return False

        if not job.output_file_id:
            print(f"  [!] Chyba: Batch je označen jako completed, ale nemá output_file_id.")
            return False

        # 2. Stáhneme obsah výstupního souboru
        print("  [>] Stahuji data z OpenAI...")
        file_response = client.files.content(job.output_file_id)
        
        # 3. Rozstříháme JSONL na jednotlivé JSONy
        lines = file_response.text.strip().split('\n')
        print(f"  [>] Zpracovávám {len(lines)} výsledků...")

        saved_count = 0
        for line in lines:
            if not line.strip(): continue
            
            data = json.loads(line)
            
            # Kontrola, zda odpověď obsahuje úspěšný výsledek
            if data.get("response", {}).get("status_code") != 200:
                print(f"  [!] Chyba u záznamu {data.get('custom_id', 'unknown')}: Status {data.get('response', {}).get('status_code')}")
                continue

            original_filename = data['custom_id']
            content_string = data['response']['body']['choices'][0]['message']['content']
            
            # Převedeme string na JSON objekt
            json_content = json.loads(content_string)
            
            # Uložení
            json_name = os.path.splitext(original_filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, json_name)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=4)
            saved_count += 1

        print(f"  [OK] Batch {batch_id} hotov. Uloženo {saved_count} souborů.")
        return True

    except Exception as e:
        print(f"  [ERR] Neočekávaná chyba u batche {batch_id}: {e}")
        return False

def main():
    if not os.path.exists(BATCH_LIST_FILE):
        print(f"Chyba: Soubor {BATCH_LIST_FILE} nebyl nalezen.")
        return

    with open(BATCH_LIST_FILE, "r") as f:
        # Načteme ID a zbavíme se mezer/nových řádků
        batch_ids = [line.strip() for line in f if line.strip()]

    print(f"Nalezeno {len(batch_ids)} ID ke kontrole.")
    
    total_completed = 0
    for bid in batch_ids:
        if download_batch_results(bid):
            total_completed += 1

    print("\n" + "="*40)
    print(f"HROMADNÉ STAHOVÁNÍ DOKONČENO")
    print(f"Úspěšně zpracováno batchů: {total_completed}/{len(batch_ids)}")
    print(f"Výsledky najdeš v: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()
