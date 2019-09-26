# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 5
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
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
# model_name = "mobilenet_v1"
model_name = "mobilenet_v2"
