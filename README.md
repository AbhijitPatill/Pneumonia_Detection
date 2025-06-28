# 🩺 Pneumonia Detection Using MobileNetV2

A deep learning project to detect pneumonia from chest X-ray images using transfer learning with MobileNetV2. Built with TensorFlow and trained on a medical X-ray dataset.

---

## 🔧 Tech Stack

- Python 🐍
- TensorFlow / Keras
- MobileNetV2 (pretrained)
- Matplotlib / Seaborn
- NumPy / Sklearn
- ImageDataGenerator for preprocessing

---

## 📂 Dataset

The dataset used is [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.  
It contains **5,863 chest X-ray images** divided into two classes:

- ✅ Normal
- 🦠 Pneumonia

---

## 🧠 Model

- Transfer learning using **MobileNetV2**
- Last few layers customized and fine-tuned
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Accuracy achieved: **~96%**

📥 Download the trained model:  
[Download model (.h5)](# 🩺 Pneumonia Detection Using MobileNetV2

A deep learning project to detect pneumonia from chest X-ray images using transfer learning with MobileNetV2. Built with TensorFlow and trained on a medical X-ray dataset.

---

## 🔧 Tech Stack

- Python 🐍
- TensorFlow / Keras
- MobileNetV2 (pretrained)
- Matplotlib / Seaborn
- NumPy / Sklearn
- ImageDataGenerator for preprocessing

---

## 📂 Dataset

The dataset used is [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.  
It contains **5,863 chest X-ray images** divided into two classes:

- ✅ Normal
- 🦠 Pneumonia

---

## 🧠 Model

- Transfer learning using **MobileNetV2**
- Last few layers customized and fine-tuned
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Accuracy achieved: **~96%**

📥 Download the trained model:  
[Download model (.h5)](https://drive.google.com/file/d/1Gt4pJ8N_5HjStZs9mVSXCfeG1TCl0clS/view?usp=drive_link)

---

## 🧪 Results

### 🔹 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### 🔹 Sample Prediction

![Sample Prediction](sample_prediction.png)

---

## ▶️ Usage

To predict a single image:

```python
from tensorflow.keras.models import load_model
from your_script import predict_image

model = load_model('mobilenet_pneumonia_model.h5')
predict_image("path/to/image.jpeg", model)
)

---

## 🧪 Results

### 🔹 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### 🔹 Sample Prediction

![Sample Prediction](sample_prediction.png)

---

## ▶️ Usage

To predict a single image:

```python
from tensorflow.keras.models import load_model
from your_script import predict_image

model = load_model('mobilenet_pneumonia_model.h5')
predict_image("path/to/image.jpeg", model)
