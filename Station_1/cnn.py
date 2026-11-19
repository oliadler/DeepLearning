import keras
from keras import layers
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

keras.utils.set_random_seed(42)

TRAIN_PFAD = "data/sharks_split/train"
VAL_PFAD = "data/sharks_split/val"
TEST_PFAD = "data/sharks_split/test"

PICTURE_SIZE = (224,224)
BATCH_SIZE = 32
CUTOUT_SIDE = 56
EPOCHS = 60
USE_CUTOUT = True
USE_AUGMIX = False



augmix = keras.layers.AugMix(value_range=(0, 255))

# aufteilen der Hai Daten
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



#  testweise 
""" for images, labels in training_data.take(1):
    img = images[32]
    plt.figure(figsize=(10, 10))
    for i in range(9):
        neues_bild = augmentation(img[None, ...], training=True)[0]
        plt.subplot(3, 3, i + 1)
        plt.imshow(neues_bild.numpy().astype("uint8"))
        plt.axis("off")
    plt.show() """

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

#testweise 
""" for bilder, labels in training_data.take(1):
    bild = bilder[43].numpy()
    s = 56
    img = cutout(bild, s)
    plt.subplot(1,2,1)
    plt.imshow(bild.astype("uint8"))
    plt.subplot(1,2,2)
    plt.imshow(img.astype("uint8"))
    plt.show()
 """


#testweise
""" 
for bilder, l in training_data.take(1):
    bild = bilder[22].numpy()
    plt.subplot(1,4,1)
    plt.imshow(bild.astype("uint8"))
    plt.title("ORIGINAL")
    augmix = keras.layers.AugMix(value_range=(0, 255))
    for i in range(0,3):
        neues_bild = augmix(bild[None, ...], training=True)[0]   
        plt.subplot(1,4,i+2)
        plt.imshow(neues_bild.numpy().astype("uint8"))
    plt.show() """


def augment_image(image, label):
    if USE_AUGMIX:
        image = augmix(image, training=True)
    if USE_CUTOUT:
        image = cutout_tf(image)
    return image,label
    
if USE_AUGMIX or USE_CUTOUT:   
    training_data = training_data.unbatch().map(augment_image).batch(BATCH_SIZE)

# testweise
""" for images, labels in training_data.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.savefig("augmentation_beispiele.png", dpi=200)
    plt.show() """

mobile_net = keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=PICTURE_SIZE + (3,),
    include_top=False,
)
mobile_net.trainable = False

print(mobile_net.output_shape)
print(mobile_net.count_params())
print(len(mobile_net.layers))

""" (None, 7, 7, 1280)
2257984
154 """

inputs = keras.Input(shape=PICTURE_SIZE + (3,))
x = augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = mobile_net(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
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
    keras.callbacks.ModelCheckpoint("best_cnn.keras", save_best_only=True),
]
start_time = time.time()
history = model.fit(
    training_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)
training_time = time.time() - start_time
print(f"Training hat {training_time:.0f} Sekunden gedauert")

model.save("cnn_final_cutout.keras")

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

plt.savefig("lernkurven_cnn_without_cutout_03_drop.png", dpi=200)
plt.show()

val_loss, val_accuracy = model.evaluate(val_data)
print(f"Beste Validierungs-Accuracy: {val_accuracy:.1%}")


# Lauf	AugMix + Cutout	Epochen	val_accuracy	Trainingszeit.         Dropout  Test    BATCH
# 1	    ja	                60	    67.1 %	        462 s underfitting. 0.3               64
# 2     nein	            34	    72.2 %	        226 s overfitting   0.3               64
# 3     ja       nein       60      67.2 %        450 s                 0.3               64
# 4     nein     ja         44      74.1 %        282 s                 0.3     76.2 %    64
# 5     nein     ja         45      69.4 %       277 s                 0.5               64
# 6     nein     ja         44        73,5 %            298 s           0.3           0.3     76,2 %    32

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
plt.savefig("confusion_matrix_cnn.png", dpi=200)
plt.show()
