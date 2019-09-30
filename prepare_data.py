import tensorflow as tf
import pathlib
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, \
    BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord
from parse_tfrecord import get_parsed_dataset


def load_and_preprocess_image(img_tensor):
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


def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count


def generate_datasets():
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)

    # read the dataset in the form of batch
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
