import keras
from keras import layers
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

keras.utils.set_random_seed(42)

TRAIN_PFAD = "data/sharks_split/train"
VAL_PFAD = "data/sharks_split/val"
TEST_PFAD = "data/sharks_split/test"

PICTURE_SIZE = (224,224)
BATCH_SIZE = 64
CUTOUT_SIDE = 56

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
    augmix_img = augmix(image, training=True)
    cutout_img = cutout_tf(augmix_img)
    return cutout_img,label
    
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
x = augmentation(training_data)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = mobile_net(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(_____, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()