import tensorflow as tf
from Backend.config import TRAIN_DIR, TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def get_data_generators():
    """
    Creates and returns data generators for training, validation, and testing.
    Uses 80-20 split on the training folder for train and val.
    """
    # Training Data Generator with Augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    # Testing Data Generator (only rescaling)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Load Training Data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Load Validation Data
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Load Testing Data
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
