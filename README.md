# Plant Leaf Disease Detection using CNN

A deep learning project that classifies plant leaf images into three categories — **Healthy**, **Powdery Mildew**, and **Rust** — using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## Overview

Plant diseases cause significant agricultural losses worldwide. Early and accurate detection can help farmers take timely action. This project uses image classification via CNN to automatically identify the health condition of plant leaves from photos.

**Classes:**
- `Healthy` — No disease detected
- `Powdery` — Powdery mildew infection
- `Rust` — Rust fungal infection

---

## Dataset Structure

```
Plant disease recognition dataset/
├── Train/
│   ├── Healthy/     (438 images)
│   ├── Powdery/     (410 images)
│   └── Rust/        (414 images)
├── Test/
│   ├── Healthy/     (50 images)
│   ├── Powdery/     (50 images)
│   └── Rust/        (50 images)
└── Validation/
    ├── Healthy/     (20 images)
    ├── Powdery/     (20 images)
    └── Rust/        (20 images)
```

**Total: 1,262 training images | 150 test images | 60 validation images**

---

## Model Architecture

A custom CNN built using Keras Sequential API:

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 64 units, ReLU |
| Dense (output) | 3 units, Softmax |

- **Input size:** 225 × 225 × 3
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Metric:** Accuracy

---

## Training Results

| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 38.2% | 71.7% |
| 2 | 70.6% | 83.3% |
| 3 | 89.3% | 75.0% |
| 4 | 89.5% | 91.7% |
| 5 | 92.9% | 90.0% |

Model achieved **~92.9% training accuracy** and **~90% validation accuracy** in just 5 epochs.

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- Seaborn
- Google Colab (training environment)

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install dependencies
```bash
pip install tensorflow numpy pillow matplotlib seaborn
```

### 3. Run in Google Colab (recommended)
Upload the notebook to Colab and mount your Google Drive with the dataset placed at:
```
MyDrive/Artificial_Intelligence_and_Machine_Learning/Practical_materials/Plant disease recognition dataset/
```

### 4. Predict on a new image
```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

x = preprocess_image('your_leaf_image.jpg')
predictions = model.predict(x)

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
print("Predicted:", labels[np.argmax(predictions)])
```

---

## 📷 Sample Prediction

Input: Rust-infected leaf image  
Output: `Rust` (confidence: 98.5%)

---


---


---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
