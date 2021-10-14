import tensorflow as tf
from configuration import NUM_CLASSES

import tensorflow.keras as nn


class SeparableConv2D(nn.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same"):
        super(SeparableConv2D, self).__init__()
        self.x1 = nn.layers.SeparableConv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding)
        self.x2 = nn.layers.BatchNormalization()
        self.x3 = nn.layers.ReLU()

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.x1(inputs)
        x = self.x2(x, training=training)
        x = self.x3(x)
        return x


class MobileNetV1(tf.keras.Model):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.separable_conv_1 = SeparableConv2D(filters=64,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.separable_conv_2 = SeparableConv2D(filters=128,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="same")
        self.separable_conv_3 = SeparableConv2D(filters=128,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.separable_conv_4 = SeparableConv2D(filters=256,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="same")
        self.separable_conv_5 = SeparableConv2D(filters=256,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.separable_conv_6 = SeparableConv2D(filters=512,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="same")

        self.separable_conv_7 = SeparableConv2D(filters=512,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.separable_conv_8 = SeparableConv2D(filters=512,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.separable_conv_9 = SeparableConv2D(filters=512,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same")
        self.separable_conv_10 = SeparableConv2D(filters=512,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding="same")
        self.separable_conv_11 = SeparableConv2D(filters=512,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding="same")

        self.separable_conv_12 = SeparableConv2D(filters=1024,
                                                 kernel_size=(3, 3),
                                                 strides=2,
                                                 padding="same")
        self.separable_conv_13 = SeparableConv2D(filters=1024,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding="same")

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                                         strides=1)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.separable_conv_1(x, training=training)
        x = self.separable_conv_2(x, training=training)
        x = self.separable_conv_3(x, training=training)
        x = self.separable_conv_4(x, training=training)
        x = self.separable_conv_5(x, training=training)
        x = self.separable_conv_6(x, training=training)
        x = self.separable_conv_7(x, training=training)
        x = self.separable_conv_8(x, training=training)
        x = self.separable_conv_9(x, training=training)
        x = self.separable_conv_10(x, training=training)
        x = self.separable_conv_11(x, training=training)
        x = self.separable_conv_12(x, training=training)
        x = self.separable_conv_13(x, training=training)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x

    def __repr__(self):
        return "MobileNetV1"
