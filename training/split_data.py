import os
import json
import random
import shutil

random.seed(42)

IMAGE_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/images_resized'
JSONS_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/jsons_resized'

TRAIN_IMAGE_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/images_train'
TEST_IMAGE_DIR  = '/storage/brno2/home/xnehez01/KNN_project/data/images_test'
TRAIN_JSON_DIR  = '/storage/brno2/home/xnehez01/KNN_project/data/jsons_train'
TEST_JSON_DIR   = '/storage/brno2/home/xnehez01/KNN_project/data/jsons_test'

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_JSON_DIR, exist_ok=True)
os.makedirs(TEST_JSON_DIR, exist_ok=True)

# Najdi vsetky pary
json_files = os.listdir(JSONS_DIR)
image_files = os.listdir(IMAGE_DIR)

pairs = []
for filename in json_files:
    if not filename.endswith(".json"):
        continue
    stem = filename.replace(".json", "")
    if stem + ".jpg" in image_files:
        pairs.append(stem)

print(f"Celkovo parov: {len(pairs)}")

random.shuffle(pairs)
split = int(0.9 * len(pairs))
train_pairs = pairs[:split]
test_pairs  = pairs[split:]

print(f"Train: {len(train_pairs)}")
print(f"Test:  {len(test_pairs)}")

# Kopíruj train
for stem in train_pairs:
    shutil.copy(os.path.join(IMAGE_DIR, stem + ".jpg"), os.path.join(TRAIN_IMAGE_DIR, stem + ".jpg"))
    shutil.copy(os.path.join(JSONS_DIR, stem + ".json"), os.path.join(TRAIN_JSON_DIR, stem + ".json"))

# Kopíruj test
for stem in test_pairs:
    shutil.copy(os.path.join(IMAGE_DIR, stem + ".jpg"), os.path.join(TEST_IMAGE_DIR, stem + ".jpg"))
    shutil.copy(os.path.join(JSONS_DIR, stem + ".json"), os.path.join(TEST_JSON_DIR, stem + ".json"))

print("Hotovo!")