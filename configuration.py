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
# 0: mobilenet_v1, 1: mobilenet_v2, 2: mobilenet_v3_large, 3: mobilenet_v3_small
# 4: efficient_net_b0, 5: efficient_net_b1, 6: efficient_net_b2, 7: efficient_net_b3
# 8: efficient_net_b4, 9: efficient_net_b5, 10: efficient_net_b6, 11: efficient_net_b7
# 12: ResNeXt50, 13: ResNeXt101
# 14: InceptionV4

# EfficientNets:
# b0 = (1.0, 1.0, 224, 0.2)
# b1 = (1.0, 1.1, 240, 0.2)
# b2 = (1.1, 1.2, 260, 0.3)
# b3 = (1.2, 1.4, 300, 0.3)
# b4 = (1.4, 1.8, 380, 0.4)
# b5 = (1.6, 2.2, 456, 0.4)
# b6 = (1.8, 2.6, 528, 0.5)
# b7 = (2.0, 3.1, 600, 0.5)
model_index = 14

