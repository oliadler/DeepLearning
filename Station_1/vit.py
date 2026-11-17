import keras
from keras import layers
from matplotlib import pyplot as plt
import keras_hub as kerashub
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
 

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
converter = kerashub.layers.ViTImageConverter.from_preset("vit_base_patch16_224_imagenet")

vit_base.trainable = False

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
    keras.callbacks.ModelCheckpoint("best_vit.keras", save_best_only=True),
]