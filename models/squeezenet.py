import tensorflow as tf
from configuration import NUM_CLASSES


class FireModule(tf.keras.layers.Layer):
    def __init__(self, s1, e1, e3):
        super(FireModule, self).__init__()
        self.squeeze_layer = tf.keras.layers.Conv2D(filters=s1,
                                                    kernel_size=(1, 1),
                                                    strides=1,
                                                    padding="same")
        self.expand_1x1 = tf.keras.layers.Conv2D(filters=e1,
                                                 kernel_size=(1, 1),
                                                 strides=1,
                                                 padding="same")
        self.expand_3x3 = tf.keras.layers.Conv2D(filters=e3,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding="same")

    def call(self, inputs, **kwargs):
        x = self.squeeze_layer(inputs)
        x = tf.nn.relu(x)
        y1 = self.expand_1x1(x)
        y1 = tf.nn.relu(y1)
        y2 = self.expand_3x3(x)
        y2 = tf.nn.relu(y2)
        return tf.concat(values=[y1, y2], axis=-1)


class SqueezeNet(tf.keras.Model):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=96,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                  strides=2)
        self.fire2 = FireModule(s1=16, e1=64, e3=64)
        self.fire3 = FireModule(s1=16, e1=64, e3=64)
        self.fire4 = FireModule(s1=32, e1=128, e3=128)
        self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                  strides=2)
        self.fire5 = FireModule(s1=32, e1=128, e3=128)
        self.fire6 = FireModule(s1=48, e1=192, e3=192)
        self.fire7 = FireModule(s1=48, e1=192, e3=192)
        self.fire8 = FireModule(s1=64, e1=256, e3=256)
        self.maxpool8 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                  strides=2)
        self.fire9 = FireModule(s1=64, e1=256, e3=256)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.conv10 = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                             kernel_size=(1, 1),
                                             strides=1,
                                             padding="same")
        self.avgpool10 = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        x = self.dropout(x, training=training)
        x = self.conv10(x)
        x = self.avgpool10(x)

        return tf.nn.softmax(x)

