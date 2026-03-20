import os

# Project Roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'Training')
TEST_DIR = os.path.join(DATASET_DIR, 'Testing')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Classes
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)
