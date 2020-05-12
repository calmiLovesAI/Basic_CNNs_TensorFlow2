import tensorflow as tf
import os

from configuration import save_model_dir, test_image_dir
from train import get_model
from prepare_data import load_and_preprocess_image


def get_class_id(image_root):
    id_cls = {}
    for i, item in enumerate(os.listdir(image_root)):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    return id_cls


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model()
    model.load_weights(filepath=save_model_dir)

    image_raw = tf.io.read_file(filename=test_image_dir)
    image_tensor = load_and_preprocess_image(image_raw)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    pred = model(image_tensor, training=False)
    idx = tf.math.argmax(pred, axis=-1).numpy()[0]

    id_cls = get_class_id("./original_dataset")

    print("The predicted category of this picture is: {}".format(id_cls[idx]))