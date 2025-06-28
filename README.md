# Pneumonia Detection 🩺🧠

This project uses **MobileNetV2** with transfer learning to classify chest X-ray images as **Pneumonia** or **Normal**.

---

## 📂 Dataset

The dataset is divided into:
- `train/`
- `val/`
- `test/`

It contains X-ray images labeled as `NORMAL` or `PNEUMONIA`.

---

## ⚙️ Model

- Pretrained on ImageNet (`MobileNetV2`)
- Fine-tuned on the pneumonia dataset
- Final layer: Sigmoid activation (binary classification)
- Saved as: `mobilenet_pneumonia_model.h5`

[📥 Download model (.h5)](https://drive.google.com/your-link-here) <!-- Replace with real link -->

---

## 🧪 Results

### 📊 Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### 🩻 Sample Prediction
![Sample Output](sample_prediction.png)

---

## 📈 Accuracy

Achieved **~96% accuracy** on the test dataset.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python pneumonia_detection.py
