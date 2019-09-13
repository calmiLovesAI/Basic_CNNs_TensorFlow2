import tensorflow as tf
from config import NUM_CLASSES

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, expansion_factor, stride):
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
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
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.linear = tf.keras.layers.Activation(tf.keras.activations.linear)

    def call(self, inputs):
        if self.stride == 1 and self.input_channels == self.output_channels:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = tf.nn.relu6(x)
            x = self.dwconv(x)
            x = self.bn2(x)
            x = tf.nn.relu6(x)
            x = self.conv2(x)
            x = self.bn3(x)
            x = self.linear(x)
            x = tf.keras.layers.add([x, inputs])
            return x
        else:
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


def build_bottleneck(t, in_channel_num, out_channel_num, n, s):
    # t : expansion factor
    # s : stride
    # n : repeat times
    bottleneck = tf.keras.Sequential()
    for i in range(n):
        if i == 0:
            bottleneck.add(BottleNeck(input_channels=in_channel_num,
                                      output_channels=out_channel_num,
                                      expansion_factor=t,
                                      stride=s))
        else:
            bottleneck.add(BottleNeck(input_channels=out_channel_num,
                                      output_channels=out_channel_num,
                                      expansion_factor=t,
                                      stride=1))

    return bottleneck


class MobileNet_V2(tf.keras.Model):
    def __init__(self):
        super(MobileNet_V2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bottleneck_1 = build_bottleneck(t=1,
                                             in_channel_num=32,
                                             out_channel_num=16,
                                             n=1,
                                             s=1)
        self.bottleneck_2 = build_bottleneck(t=6,
                                             in_channel_num=16,
                                             out_channel_num=24,
                                             n=2,
                                             s=2)
        self.bottleneck_3 = build_bottleneck(t=6,
                                             in_channel_num=24,
                                             out_channel_num=32,
                                             n=3,
                                             s=2)
        self.bottleneck_4 = build_bottleneck(t=6,
                                             in_channel_num=32,
                                             out_channel_num=64,
                                             n=4,
                                             s=2)
        self.bottleneck_5 = build_bottleneck(t=6,
                                             in_channel_num=64,
                                             out_channel_num=96,
                                             n=3,
                                             s=1)
        self.bottleneck_6 = build_bottleneck(t=6,
                                             in_channel_num=96,
                                             out_channel_num=160,
                                             n=3,
                                             s=2)
        self.bottleneck_7 = build_bottleneck(t=6,
                                             in_channel_num=160,
                                             out_channel_num=320,
                                             n=1,
                                             s=1)

        self.conv2 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))
        self.conv3 = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            activation=tf.keras.activations.softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)

        return x

