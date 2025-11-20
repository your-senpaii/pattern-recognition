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

train_dir = "training/"   # your training folder
test_dir = "testing/"     # your testing folder

img_size = (128, 128)
batch_size = 32  # Increased from 16 to better utilize GPU

# Training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

# Testing data - only rescaling, no augmentation
test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",   # multi-class labels
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False
)

print("Class indices:", train_data.class_indices)
print(f"Training samples: {train_data.samples}")
print(f"Testing samples: {test_data.samples}")

# ======================================
# 2. BUILD CNN MODEL (MULTI-CLASS)
# ======================================

num_classes = len(train_data.class_indices)  # Auto-detect number of classes

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
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================================
# 5. CONFUSION MATRIX
# ======================================

# Reset test_data generator to start from beginning
test_data.reset()

true_labels = test_data.classes
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=train_data.class_indices.keys(),
            yticklabels=train_data.class_indices.keys(),
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, 
                            target_names=train_data.class_indices.keys()))

# ======================================
# 6. Show one sample prediction from each class
# ======================================

# Get class names
class_names = list(train_data.class_indices.keys())

# Find one sample from each class
sample_indices = []
for class_idx in range(num_classes):
    # Find first occurrence of this class in test data
    for i, label in enumerate(true_labels):
        if label == class_idx:
            sample_indices.append(i)
            break

# Display samples
plt.figure(figsize=(20, 5))
test_data.reset()

for idx, sample_idx in enumerate(sample_indices):
    # Get image and labels
    batch_idx = sample_idx // batch_size
    img_idx = sample_idx % batch_size
    
    # Navigate to the correct batch
    test_data.reset()
    for _ in range(batch_idx + 1):
        imgs, labels = next(test_data)
    
    img = imgs[img_idx]
    true_label = true_labels[sample_idx]
    pred_label = pred_labels[sample_idx]
    
    plt.subplot(1, num_classes, idx + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Class: {class_names[true_label]}\nPredicted: {class_names[pred_label]}", 
              fontsize=12,
              color='green' if true_label == pred_label else 'red',
              weight='bold')

plt.suptitle("Sample Prediction from Each Class", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# ======================================
# 7. SAVE THE MODEL (Optional)
# ======================================

# Uncomment to save the model
# model.save('image_classifier_model.h5')
# print("\nModel saved as 'image_classifier_model.h5'")