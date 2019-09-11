# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 5
image_height = 224
image_width = 224
channels = 3
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
model = "mobilenet_v1"
# model = "mobilenet_v2"
