import keras

keras.utils.set_random_seed(42)

TRAIN_PFAD = "data/sharks_split/train"
VAL_PFAD = "data/sharks_split/val"
TEST_PFAD = "data/sharks_split/test"

PICTURE_SIZE = (224,224)
BATCH_SIZE = 64

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

for batch in training_data.take(1):
    images, labels = batch
    print(images.shape)
    print(labels.shape)