import tensorflow as tf
import pathlib
from config import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord
from parse_tfrecord import get_parsed_dataset

def load_and_preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.io.decode_jpeg(img_raw, channels=CHANNELS)
    # resize
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = tf.dtypes.cast(img_tensor, tf.dtypes.float32)
    # normalization
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((index, label) for label, index in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_dataset(dataset_tfrecord):
    # all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # load the dataset and preprocess images
    # image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    # label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    # dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    # image_count = len(all_image_path)
    #
    # return dataset, image_count
    parsed_dataset = get_parsed_dataset(dataset_tfrecord)
    image_paths = []
    labels = []
    for features in parsed_dataset:
        image_path = features['image_path'].numpy()
        image_path_string = str(image_path, encoding='utf-8')
        label = features['label'].numpy()
        image_paths.append(image_path_string)
        labels.append(label)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(image_paths)
    return dataset, image_count



def generate_datasets():
    train_dataset, train_count = get_dataset(dataset_tfrecord=train_tfrecord)
    valid_dataset, valid_count = get_dataset(dataset_tfrecord=valid_tfrecord)
    test_dataset, test_count = get_dataset(dataset_tfrecord=test_tfrecord)

    # read the dataset in the form of batch
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
