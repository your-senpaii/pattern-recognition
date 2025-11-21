import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import visualization as vs 

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
epochs = 5 
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs
)

vs.plot_metrics(history)

test_data.reset()
true_labels = test_data.classes
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

cm = confusion_matrix(true_labels, pred_labels)
vs.plot_confusion_matrix(cm, list(train_data.class_indices.keys()))

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, 
                            target_names=train_data.class_indices.keys()))

vs.visualize_sample_predictions(test_data, true_labels, pred_labels, num_classes, batch_size)