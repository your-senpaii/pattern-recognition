import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

train_dir = "training/" 
test_dir = "testing/"    

img_size = (128, 128)
batch_size = 32  

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",   
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

num_classes = len(train_data.class_indices) 

model = models.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
epochs = 4

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs
)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

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

class_names = list(train_data.class_indices.keys())

sample_indices = []
for class_idx in range(num_classes):
    all_indices_for_class = np.where(true_labels == class_idx)[0]
    
    if len(all_indices_for_class) > 0:
        random_index = np.random.choice(all_indices_for_class)
        sample_indices.append(random_index)
    else:
        print(f"Warning: No samples found for class index {class_idx}")

plt.figure(figsize=(20, 5))
test_data.reset()

for idx, sample_idx in enumerate(sample_indices):
    batch_idx = sample_idx // batch_size
    img_idx = sample_idx % batch_size
    
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