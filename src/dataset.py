import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
DATA_DIR = "data_processed"

LABELS = {
    "Normal": 0,
    "LG": 1,
    "HG": 2
}

def load_dataset():
    images = []
    labels = []

    for class_name, label in LABELS.items():
        class_dir = os.path.join(DATA_DIR, class_name)

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(".bmp"):
                img_path = os.path.join(class_dir, filename)

                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = img / 255.0

                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


def get_train_test_data(test_size=0.2):
    X, y = load_dataset()
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
