import tensorflow as tf


def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6


def h_swish(x):
    return x * h_sigmoid(x)


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = h_sigmoid(branch)
        output = inputs * branch
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self):
        super(BottleNeck, self).__init__()
        pass

    def call(self, inputs, training=None, **kwargs):
        pass
