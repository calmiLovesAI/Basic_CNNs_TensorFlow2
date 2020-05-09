import tensorflow as tf

from configuration import NUM_CLASSES
from models.group_convolution import GroupConv2D


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = tf.nn.sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = tf.keras.layers.multiply(inputs=[inputs, branch])
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters,
                                      output_channels=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding="same",
                                      groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=2 * filters)

        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.se(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


class SEResNeXt(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality):
        super(SEResNeXt, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.block1 = SEResNeXt.__make_layer(filters=128,
                                             strides=1,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[0])
        self.block2 = SEResNeXt.__make_layer(filters=256,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[1])
        self.block3 = SEResNeXt.__make_layer(filters=512,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[2])
        self.block4 = SEResNeXt.__make_layer(filters=1024,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[3])
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    @staticmethod
    def __make_layer(filters, strides, groups, repeat_num):
        block = tf.keras.Sequential()
        block.add(BottleNeck(filters=filters,
                             strides=strides,
                             groups=groups))
        for _ in range(1, repeat_num):
            block.add(BottleNeck(filters=filters,
                                 strides=1,
                                 groups=groups))

        return block

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x


def SEResNeXt50():
    return SEResNeXt(repeat_num_list=[3, 4, 6, 3], cardinality=32)


def SEResNeXt101():
    return SEResNeXt(repeat_num_list=[3, 4, 23, 3], cardinality=32)