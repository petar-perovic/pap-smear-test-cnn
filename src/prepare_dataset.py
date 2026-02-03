import os
import shutil

RAW_DATA_DIR = "data/raw/smear2005/New database pictures"
PROCESSED_DATA_DIR = "data_processed"

CLASS_MAPPING = {
    "normal_superficiel": "Normal",
    "normal_columnar": "Normal",
    "normal_intermediate": "Normal",
    "light_dysplastic": "LG",
    "moderate_dysplastic": "HG",
    "severe_dysplastic": "HG",
    "carcinoma_in_situ": "HG"
}

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

for new_class in set(CLASS_MAPPING.values()):
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, new_class), exist_ok=True)

for old_class, new_class in CLASS_MAPPING.items():
    source_dir = os.path.join(RAW_DATA_DIR, old_class)
    target_dir = os.path.join(PROCESSED_DATA_DIR, new_class)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".bmp"):
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)

            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

print("Dataset uspješno pripremljen.")
