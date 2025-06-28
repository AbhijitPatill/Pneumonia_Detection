# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Disable OneDNN Optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print(" All Libraries Imported!")

#  Dataset Paths
train_dir = r"C:\Users\91976\OneDrive\Pictures\Desktop\Pneumonia detection\train"
test_dir = r"C:\Users\91976\OneDrive\Pictures\Desktop\Pneumonia detection\test"
val_dir = r"C:\Users\91976\OneDrive\Pictures\Desktop\Pneumonia detection\val"

# Image Dimensions
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

#  Data Generators
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
val_data = test_val_gen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
test_data = test_val_gen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

print(" Data Loaded Successfully!")

#  Model Path
model_path = "mobilenet_pneumonia_model.h5"

# Define or Load Model
if os.path.exists(model_path):
    model = load_model(model_path)
    print(" Pretrained MobileNetV2 Model Loaded!")
else:
    print("🔧 Training New MobileNetV2 Model...")

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    # Clean memory
    gc.collect()
    tf.keras.backend.clear_session()

    # Train the Model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stopping, checkpoint]
    )

    # Plot Accuracy & Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Model Loss")
    plt.show()

# Evaluate on Test Set
print("🧪 Evaluating on Test Data...")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")

# 🧾 Classification Report
y_pred = model.predict(test_data)
y_pred_class = (y_pred > 0.5).astype("int32")
y_true = test_data.classes

print("\n Classification Report:")
print(classification_report(y_true, y_pred_class, target_names=["Normal", "Pneumonia"]))

sns.heatmap(confusion_matrix(y_true, y_pred_class), annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#  Predict Function for Single Image
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.6:
        result = "Pneumonia Detected "
    else:
        result = "Normal X-ray "

    print(f"Prediction: {result}")
    plt.imshow(img)
    plt.title(result)
    plt.axis("off")
    plt.show()

#  Predict on Sample Image
sample_image = r"C:\Users\91976\OneDrive\Pictures\Desktop\Pneumonia detection\test\PNEUMONIA\person1_virus_7.jpeg"
predict_image(sample_image, model)
