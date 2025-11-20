import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ======================================
# 1. DATASET LOADING + AUGMENTATION
# ======================================

data_dir = "dataset/"   # your dataset folder

img_size = (128, 128)
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.30,  # 70% train, 30% test
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",   # multi-class labels
    subset="training"
)

test_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_data.class_indices)

# ======================================
# 2. BUILD CNN MODEL (MULTI-CLASS)
# ======================================

num_classes = 4  # flower, astronaut, guillotine, cube

model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================
# 3. TRAIN THE MODEL
# ======================================

epochs = 15

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs
)

# ======================================
# 4. PLOT TRAINING CURVES
# ======================================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.show()

# ======================================
# 5. CONFUSION MATRIX
# ======================================

true_labels = test_data.classes
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_data.class_indices.keys(),
            yticklabels=train_data.class_indices.keys(),
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

# ======================================
# 6. Show predictions for 3 random images
# ======================================

import random

indices = random.sample(range(len(pred_labels)), 3)

for i in indices:
    img, label = test_data[i][0][0], true_labels[i]
    pred = pred_labels[i]

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Actual: {label},  Predicted: {pred}")
    plt.show()
