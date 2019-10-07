import tensorflow as tf
from models.inception_modules import BasicConv2D, Conv2DLinear


class Stem(tf.keras.layers.Layer):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = BasicConv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=2,
                                 padding="valid")
        self.conv2 = BasicConv2D(filters=32,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding="valid")
        self.conv3 = BasicConv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding="same")
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                 strides=2,
                                                 padding="valid")
        self.conv4 = BasicConv2D(filters=80,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
        self.conv5 = BasicConv2D(filters=192,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding="valid")
        self.conv6 = BasicConv2D(filters=256,
                                 kernel_size=(3, 3),
                                 strides=2,
                                 padding="valid")

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.maxpool(x)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)

        return x


class InceptionResnetA(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionResnetA, self).__init__()
        self.b1_conv = BasicConv2D(filters=32,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.b2_conv1 = BasicConv2D(filters=32,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b2_conv2 = BasicConv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")
        self.b3_conv1 = BasicConv2D(filters=32,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b3_conv2 = BasicConv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")
        self.b3_conv3 = BasicConv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")
        self.conv = Conv2DLinear(filters=256,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_conv(inputs, training=training)
        b2 = self.b2_conv1(inputs, training=training)
        b2 = self.b2_conv2(b2, training=training)
        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)
        b3 = self.b3_conv3(b3, training=training)

        x = tf.concat(values=[b1, b2, b3], axis=-1)

        output = tf.keras.layers.add([x, inputs])
        return tf.nn.relu(output)