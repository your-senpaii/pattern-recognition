import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ============================
# 1. Image Data Generators
# ============================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    "training/",
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    shuffle=True,
)

test_data = test_datagen.flow_from_directory(
    "testing/",
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    shuffle=False,
)

# ============================
# 2. SIMPLE CNN MODEL
# ============================

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ============================
# 3. TRAIN MODEL
# ============================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15
)

# ============================
# 4. EVALUATION
# ============================

loss, acc = model.evaluate(test_data)
print("Test accuracy:", acc)
