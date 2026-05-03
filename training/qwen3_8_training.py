import torch
import os
from PIL import Image
from transformers import set_seed
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from typing import List, Dict, Any
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

#used https://www.datacamp.com/tutorial/fine-tuning-qwen3-vl-8b
set_seed(14)

MODEL_PATH = 'PATH'
IMAGE_DIR = 'PATH'
JSONS_DIR = 'PATH'

#T32 formating 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
# print("bf16 supported:", torch.cuda.is_bf16_supported())

#nacitanie modelu
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialized in extracting tables of contents "
    "from scanned book pages. "
    "For each entry in the table of contents return a JSON object with these fields: "
    "name (chapter title), "
    "chapter_number (chapter number or null), "
    "page_number (page number as string or null), "
    "description (description text or null), "
    "name_bbox ([x1, y1, x2, y2] in pixels or null), "
    "chapter_number_bbox ([x1, y1, x2, y2] in pixels or null), "
    "page_number_bbox ([x1, y1, x2, y2] in pixels or null), "
    "description_bbox ([x1, y1, x2, y2] in pixels or null), "
    "subchapters (if the entry has subchapters, list them here with the same JSON structure as parent chapters; if no subchapters exist, use empty list []). "
    "If a field is not visible, set it to null. "
    "Return only a valid JSON array, no markdown, no explanation."
)

USER_PROMPT = (
    "Extract all entries from the table of contents in this scanned book page. "
    "Return a JSON array where each element represents one entry "
    "with its text and bounding box coordinates in pixels."
)
# ensure ascii pre ascii znaky
def annotation_to_target(annotation):
    return json.dumps(annotation, ensure_ascii=False)

def load_paired_data():
    samples = []
    json_files = os.listdir(JSONS_DIR)
    image_files = os.listdir(IMAGE_DIR)

    for filename in json_files:
        if not filename.endswith(".json"):
            continue
        json_path = os.path.join(JSONS_DIR, filename)
        stem = filename.replace(".json", "")
        if stem + ".jpg" not in image_files:
            continue
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            samples.append({
                "annotation": json_data,
                "image_path": os.path.join(IMAGE_DIR, stem + ".jpg")
            })
    return samples

def build_messages(image, annotation):
    target = annotation_to_target(annotation)
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}]
        },
    ]

# build hugging face type dataset
class TOCDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image = image.convert("RGB")
        messages = build_messages(image, sample["annotation"])
        return {"messages": messages}
    

MAX_LEN = 8192

def collate_fn(batch: List[Dict[str, Any]]):
    # 1) Build full chat text (includes assistant answer)
    full_texts = [
        processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for ex in batch
    ]

    # 2) Build prompt-only text (up to user turn; generation prompt on)
    prompt_texts = [
        processor.apply_chat_template(
            ex["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in batch
    ]
    # 3) Images
    images = []
    for ex in batch:
        for msg in ex["messages"]:
            for item in msg["content"]:
                if item["type"] == "image":
                    images.append(item["image"])

    # 4) Tokenize full inputs ONCE (text + images) converted to model understandable numbers
    enc = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    input_ids = enc["input_ids"]
    pad_id = processor.tokenizer.pad_token_id
    labels = input_ids.clone()

    # 5) Compute prompt lengths with TEXT-ONLY tokenization
    prompt_ids = processor.tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False,
    )["input_ids"]

    prompt_lens = (prompt_ids != pad_id).sum(dim=1)
    
    bs, seqlen = labels.shape

    for i in range(bs):
        pl = int(prompt_lens[i].item())
        pl = min(pl, seqlen)
        labels[i, :pl] = -100

    # Mask padding positions too
    labels[labels == pad_id] = -100

    # If your processor produces pixel_values / image_grid_thw, keep them
    enc["labels"] = labels
    return enc

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)

args = SFTConfig(
    output_dir="/storage/brno2/home/xnehez01/KNN_project/finetuned/qwen3_toc_lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_steps=50,
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)

samples = load_paired_data()
dataset = TOCDataset(samples)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    peft_config=lora,
)

trainer.train()
trainer.save_model()