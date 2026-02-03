# Pap Smear Cell Classification using Convolutional Neural Networks

This repository contains a Python-based machine learning project for **automatic classification of Pap smear cell images** into three clinically relevant classes using **Convolutional Neural Networks (CNNs)**.

The project is intended for academic and educational purposes.

---

## Project Overview

The goal of this project is to classify cervical cell images from Pap smear tests into **three classes**:

* **Normal**
* **LG lesion (Low-grade lesion)**
* **HG lesion (High-grade lesion)**

The model is trained using a **CNN architecture** implemented in TensorFlow/Keras and evaluated using standard classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

---

## Original Dataset Structure

The original dataset (Herlev Pap Smear Dataset) contains **seven subclasses** organized as follows:

```
smear2005/
└── New database pictures/
    ├── normal_superficiel/
    ├── normal_columnar/
    ├── normal_intermediate/
    ├── light_dysplastic/
    ├── moderate_dysplastic/
    ├── severe_dysplastic/
    └── carcinoma_in_situ/
```

These seven subclasses are **mapped into three target classes**:

* **Normal**

  * normal_superficiel
  * normal_columnar
  * normal_intermediate

* **LG (Low-grade lesion)**

  * light_dysplastic

* **HG (High-grade lesion)**

  * moderate_dysplastic
  * severe_dysplastic
  * carcinoma_in_situ



You can download dataset:
  Herlev Pap Smear Dataset
  [https://mde-lab.aegean.gr/index.php/downloads/](https://mde-lab.aegean.gr/index.php/downloads/)
  (file: `smear2005.zip`)

* **Reference Paper:**
  [https://telkomnika.uad.ac.id/index.php/TELKOMNIKA/article/view/22440](https://telkomnika.uad.ac.id/index.php/TELKOMNIKA/article/view/22440)

Dataset should be imported in project as data/raw/smear2005/New database pictures

---

## Dataset Preparation

Dataset preprocessing and class mapping are performed using the script:

```bash
python src/prepare_dataset.py
```

This script:

* Reads images from the original dataset structure
* Maps the 7 subclasses into 3 classes (Normal, LG, HG)
* Copies images into a new processed dataset structure

### Resulting directory structure:

```
data_processed/
├── Normal/
├── LG/
└── HG/
```

## Dataset Loading and Verification

The `dataset.py` module:

* Loads images from `data_processed/`
* Resizes images to **128 × 128** pixels
* Normalizes pixel values to the range **[0, 1]**
* Prepares NumPy arrays suitable for CNN training

### Quick sanity check

You can verify that dataset loading works correctly by running:

```bash
python
```

Then inside the Python shell:

```python
from src.dataset import get_train_test_data

X_train, X_test, y_train, y_test = get_train_test_data()

print(X_train.shape)
print(y_train.shape)
```

Expected output:

```
(XXXX, 128, 128, 3)
(XXXX,)
```

Total dataset size:

```
(1834, 128, 128, 3)
```

---

## CNN Model

The CNN architecture is defined in:

```
src/model.py
```

It includes:

* Convolutional layers for feature extraction
* Pooling layers for dimensionality reduction
* Fully connected layers
* Softmax output layer for 3-class classification

---

## Model Training

Training logic is implemented in:

```
src/train.py
```

The script:

* Loads processed data using `dataset.py`
* Builds the CNN model using `model.py`
* Trains the model with **early stopping regularization**
* Saves the trained model to:

```
results/cnn_model.h5
```

---

## Environment Setup

At the moment of posting repository TensorFlow does not support the latest Python versions. Therefore, **Python 3.10** is used together with a virtual environment to avoid dependency conflicts. If your version of python supports it, you can skip this part.

### Create and activate virtual environment

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

### Install required dependencies

```bash
pip install tensorflow==2.15.0 opencv-python scikit-learn numpy
```

---

## Training the Model

From the project root directory (with virtual environment activated):

```bash
python -m src.train
```

After training, the model will be saved as:

```
results/cnn_model.h5
```

---

## Model Evaluation

To evaluate the trained model and visualize results:

### Install additional libraries

```bash
pip install matplotlib seaborn
```

### Run evaluation

```bash
python -m src.evaluate
```

This script outputs:

* Accuracy
* Precision, Recall, F1-score per class
* Confusion Matrix visualization

---



## Notes

* The dataset is **not included** in this repository.
* This project is designed for **educational and research purposes**.
* Results may vary depending on hardware and random initialization.