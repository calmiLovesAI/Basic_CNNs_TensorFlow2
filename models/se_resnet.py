import tensorflow as tf
from configuration import NUM_CLASSES


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
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=filter_num * 4)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.se(x)
        output = tf.nn.relu(tf.keras.layers.add([identity, x]))
        return output


class SEResNet(tf.keras.Model):
    def __init__(self, block_num):
        super(SEResNet, self).__init__()

        self.pre1 = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding='same')
        self.pre2 = tf.keras.layers.BatchNormalization()
        self.pre3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.pre4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2)

        self.layer1 = self._make_res_block(filter_num=64,
                                           blocks=block_num[0])
        self.layer2 = self._make_res_block(filter_num=128,
                                           blocks=block_num[1],
                                           stride=2)
        self.layer3 = self._make_res_block(filter_num=256,
                                           blocks=block_num[2],
                                           stride=2)
        self.layer4 = self._make_res_block(filter_num=512,
                                           blocks=block_num[3],
                                           stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def _make_res_block(self, filter_num, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filter_num, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BottleNeck(filter_num, stride=1))

        return res_block

    def call(self, inputs, training=None, mask=None):
        pre1 = self.pre1(inputs)
        pre2 = self.pre2(pre1, training=training)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        l1 = self.layer1(pre4, training=training)
        l2 = self.layer2(l1, training=training)
        l3 = self.layer3(l2, training=training)
        l4 = self.layer4(l3, training=training)
        avgpool = self.avgpool(l4)
        out = self.fc(avgpool)
        return out


def se_resnet_50():
    return SEResNet(block_num=[3, 4, 6, 3])


def se_resnet_101():
    return SEResNet(block_num=[3, 4, 23, 3])


def se_resnet_152():
    return SEResNet(block_num=[3, 8, 36, 3])