from __future__ import absolute_import, division, print_function
import tensorflow as tf
from configuration import CHANNELS, EPOCHS, BATCH_SIZE, save_model_dir
from prepare_data import generate_datasets, load_and_preprocess_image
from models.efficientnet import get_efficient_net
import math

def process_features(features):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = tf.io.decode_jpeg(contents=image, channels=CHANNELS)
        image_tensor_list.append(image_tensor)
    image_tensors = tf.stack(image_tensor_list, axis=0)
    images = load_and_preprocess_image(image_tensors)
    labels = features['label'].numpy()

    return images, labels

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # get the dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    # b0 = get_efficient_net(1.0, 1.0, 224, 0.2)
    # b1 = get_efficient_net(1.0, 1.1, 240, 0.2)
    # b2 = get_efficient_net(1.1, 1.2, 260, 0.3)
    # b3 = get_efficient_net(1.2, 1.4, 300, 0.3)
    # b4 = get_efficient_net(1.4, 1.8, 380, 0.4)
    # b5 = get_efficient_net(1.6, 2.2, 456, 0.4)
    # b6 = get_efficient_net(1.8, 2.6, 528, 0.5)
    # b7 = get_efficient_net(2.0, 3.1, 600, 0.5)
    model = get_efficient_net(1.2, 1.4, 300, 0.3)


    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        step = 0
        for features in train_dataset:
            step += 1
            images, labels = process_features(features)
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()))


        for features in valid_dataset:
            valid_images, valid_labels = process_features(features)
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_accuracy.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()


    # save the whole model
    tf.saved_model.save(model, save_model_dir)