import keras
from keras import layers
from matplotlib import pyplot as plt
import keras_hub as kerashub
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
 

TRAIN_PFAD = "data/sharks_split/train"
VAL_PFAD = "data/sharks_split/val"
TEST_PFAD = "data/sharks_split/test"

PICTURE_SIZE = (224,224)
BATCH_SIZE = 32
CUTOUT_SIDE = 56
EPOCHS = 1
USE_CUTOUT = True
USE_AUGMIX = False
LORA_RANK = 4

augmix = keras.layers.AugMix(value_range=(0, 255))
keras.utils.set_random_seed(42)

training_data = keras.utils.image_dataset_from_directory(
    TRAIN_PFAD,
    image_size=PICTURE_SIZE,
    batch_size=BATCH_SIZE,
    seed=42
)
val_data = keras.utils.image_dataset_from_directory(
    VAL_PFAD,
    image_size=PICTURE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_data = keras.utils.image_dataset_from_directory(
    TEST_PFAD,
    image_size=PICTURE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = training_data.class_names
count_classes = len(class_names)


augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03, fill_mode="reflect"),
    layers.RandomZoom(0.1,fill_mode="reflect"),
    layers.RandomBrightness(0.2, value_range=(0,255)),
    layers.RandomContrast(0.2),
], name="augmentation")

def cutout(img, side):
    image = img.copy()
    top = np.random.randint(0,image.shape[0] -  side + 1)
    left = np.random.randint(0,image.shape[1] - side + 1)
    image[top:top+side, left:left+side] = 0
    return image

def cutout_tf(image):
    new_image = tf.numpy_function(cutout, [image, CUTOUT_SIDE], tf.float32)
    new_image.set_shape(image.shape)
    return new_image

def augment_image(image, label):
    if USE_AUGMIX:
        image = augmix(image, training=True)
    if USE_CUTOUT:
        image = cutout_tf(image)
    return image,label
    
if USE_AUGMIX or USE_CUTOUT:   
    training_data = training_data.unbatch().map(augment_image).batch(BATCH_SIZE)

vit_base = kerashub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet")
vit_base.enable_lora(rank=LORA_RANK)
print(vit_base.default_lora_layer_names())
converter = kerashub.layers.ViTImageConverter.from_preset("vit_base_patch16_224_imagenet")

inputs = keras.Input(shape=PICTURE_SIZE + (3,))
x = augmentation(inputs)
x = converter(x)
x = vit_base(x,training=False)
x = x[:, 0, :]
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(count_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_vit_lora.keras", save_best_only=True),
]

start = time.time()
history = model.fit(
    training_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)
endtime = time.time() - start

print(f"Training hat {endtime:.0f} Sekunden gedauert")

model.save("vit_lora_final_cutout_rank4.keras")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validierung")
plt.title("Accuracy")
plt.xlabel("Epoche")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validierung")
plt.title("LOSS")
plt.xlabel("Epoche")
plt.legend()

plt.savefig("lernkurven_vit_lora_without_cutout_03_drop_rank4.png", dpi=200)
plt.show()

val_loss, val_accuracy = model.evaluate(val_data)
print(f"Beste Validierungs-Accuracy: {val_accuracy:.1%}")

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy:{test_accuracy:.1%}")

# welche bilder sind falsch?
probabilities = model.predict(test_data)
predictions = np.argmax(probabilities, axis=1)
confidence = np.max(probabilities,axis=1)
true_labels = np.concatenate([labels.numpy() for images, labels in test_data])

print(probabilities.shape)
print(predictions.shape)
print(true_labels.shape)

# falsche Bilder
for path, true, pred, conf in zip(test_data.file_paths, true_labels, predictions, confidence):
    if pred != true:
        print(f"{path}: true={class_names[true]} predicted={class_names[pred]} ({conf:.1%})")

print(classification_report(true_labels,predictions,target_names=class_names))
matrix = confusion_matrix(true_labels, predictions)

fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(ax=ax, xticks_rotation=90)
plt.tight_layout()
plt.savefig("confusion_matrix_vit_lora_rank4.png", dpi=200)
plt.show()
