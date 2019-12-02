import tensorflow as tf

from configuration import NUM_CLASSES


def channel_shuffle(feature, group):
    channel_num = feature.shape[-1]
    if channel_num % group != 0:
        raise ValueError("The group must be divisible by the shape of the last dimension of the feature.")
    x = tf.reshape(feature, shape=(-1, feature.shape[1], feature.shape[2], group, channel_num // group))
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    x = tf.reshape(x, shape=(-1, feature.shape[1], feature.shape[2], channel_num))
    return x


class ShuffleBlockS1(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(ShuffleBlockS1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding="same")
        self.dw_bn = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        branch, x = tf.split(inputs, num_or_size_splits=2, axis=-1)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dwconv(x)
        x = self.dw_bn(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        outputs = tf.concat(values=[branch, x], axis=-1)
        outputs = channel_shuffle(feature=outputs, group=2)
        return outputs


class ShuffleBlockS2(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(ShuffleBlockS2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels // 2,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding="same")
        self.dw_bn = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels - in_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.branch_dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding="same")
        self.branch_dwbn = tf.keras.layers.BatchNormalization()
        self.branch_conv = tf.keras.layers.Conv2D(filters=in_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.branch_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dwconv(x)
        x = self.dw_bn(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        branch = self.branch_dwconv(inputs)
        branch = self.branch_dwbn(branch, training=training)
        branch = self.branch_conv(branch)
        branch = self.branch_bn(branch, training=training)
        branch = tf.nn.relu(branch)

        outputs = tf.concat(values=[x, branch], axis=-1)
        outputs = channel_shuffle(feature=outputs, group=2)
        return outputs


class ShuffleNetV2(tf.keras.Model):
    def __init__(self, channel_scale):
        super(ShuffleNetV2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=2, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        self.stage1 = self._make_layer(repeat_num=4, in_channels=24, out_channels=channel_scale[0])
        self.stage2 = self._make_layer(repeat_num=8, in_channels=channel_scale[0], out_channels=channel_scale[1])
        self.stage3 = self._make_layer(repeat_num=4, in_channels=channel_scale[1], out_channels=channel_scale[2])
        self.conv5 = tf.keras.layers.Conv2D(filters=channel_scale[3], kernel_size=(1, 1), strides=1, padding="same")
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def _make_layer(self, repeat_num, in_channels, out_channels):
        block = tf.keras.Sequential()
        block.add(ShuffleBlockS2(in_channels=in_channels, out_channels=out_channels))
        for i in range(1, repeat_num):
            block.add(ShuffleBlockS1(in_channels=out_channels, out_channels=out_channels))
        return block

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool(x)
        x = self.stage1(x, training=training)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = tf.nn.relu(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def shufflenet_0_5x():
    return ShuffleNetV2(channel_scale=[48, 96, 192, 1024])


def shufflenet_1_0x():
    return ShuffleNetV2(channel_scale=[116, 232, 464, 1024])


def shufflenet_1_5x():
    return ShuffleNetV2(channel_scale=[176, 352, 704, 1024])


def shufflenet_2_0x():
    return ShuffleNetV2(channel_scale=[244, 488, 976, 2048])
