import os
import shutil
import random

source_dir = "PlantVillage"
base_dir = "dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_end = int(train_ratio * len(images))
    val_end = int((train_ratio + val_ratio) * len(images))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        split_class_dir = os.path.join(base_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in splits[split]:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copyfile(src, dst)

print("Dataset prepared successfully!")