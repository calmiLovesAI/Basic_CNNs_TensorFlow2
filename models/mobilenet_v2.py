import tensorflow as tf

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, expansion_factor, stride):
        self.stride = stride
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=input_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                                      strides=stride,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=output_channels,
                                            kernel_sise=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.linear = tf.keras.layers.Activation(tf.keras.activations.linear)

    def call(self, inputs):
        if self.stride == 1:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = tf.nn.relu6(x)
            x = self.dwconv(x)
            x = self.bn2(x)
            x = tf.nn.relu6(x)
            x = self.conv2(x)
            x = self.bn3(x)
            x = self.linear(x)
            x = tf.keras.layers.Add([x, inputs])
        elif self.stride == 2:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = tf.nn.relu6(x)
            x = self.dwconv(x)
            x = self.bn2(x)
            x = tf.nn.relu6(x)
            x = self.conv2(x)
            x = self.bn3(x)
            x = self.linear(x)

        return x

