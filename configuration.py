# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 5
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
CHANNELS = 3
save_model_dir = "saved_model/"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2

# choose a network
# 0: mobilenet_v1, 1: mobilenet_v2, 2: mobilenet_v3_large, 3: mobilenet_v3_small

model_index = 2

