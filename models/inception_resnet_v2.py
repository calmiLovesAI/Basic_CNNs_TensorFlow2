import tensorflow as tf
from models.inception_modules import Stem, ReductionA, BasicConv2D, Conv2DLinear
from configuration import NUM_CLASSES


class InceptionResNetA(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionResNetA, self).__init__()
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
        self.b3_conv2 = BasicConv2D(filters=48,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")
        self.b3_conv3 = BasicConv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")
        self.conv = Conv2DLinear(filters=384,
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
        x = self.conv(x, training=training)

        output = tf.keras.layers.add([x, inputs])
        return tf.nn.relu(output)


class InceptionResNetB(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionResNetB, self).__init__()
        self.b1_conv = BasicConv2D(filters=192,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.b2_conv1 = BasicConv2D(filters=128,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b2_conv2 = BasicConv2D(filters=160,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding="same")
        self.b2_conv3 = BasicConv2D(filters=192,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding="same")
        self.conv = Conv2DLinear(filters=1152,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_conv(inputs, training=training)
        b2 = self.b2_conv1(inputs, training=training)
        b2 = self.b2_conv2(b2, training=training)
        b2 = self.b2_conv3(b2, training=training)

        x = tf.concat(values=[b1, b2], axis=-1)
        x = self.conv(x, training=training)

        output = tf.keras.layers.add([x, inputs])

        return tf.nn.relu(output)


class InceptionResNetC(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionResNetC, self).__init__()
        self.b1_conv = BasicConv2D(filters=192,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.b2_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b2_conv2 = BasicConv2D(filters=224,
                                    kernel_size=(1, 3),
                                    strides=1,
                                    padding="same")
        self.b2_conv3 = BasicConv2D(filters=256,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding="same")
        self.conv = Conv2DLinear(filters=2144,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_conv(inputs, training=training)
        b2 = self.b2_conv1(inputs, training=training)
        b2 = self.b2_conv2(b2, training=training)
        b2 = self.b2_conv3(b2, training=training)

        x = tf.concat(values=[b1, b2], axis=-1)
        x = self.conv(x, training=training)

        output = tf.keras.layers.add([x, inputs])

        return tf.nn.relu(output)


class ReductionB(tf.keras.layers.Layer):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                    strides=2,
                                                    padding="valid")
        self.b2_conv1 = BasicConv2D(filters=256,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b2_conv2 = BasicConv2D(filters=384,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding="valid")
        self.b3_conv1 = BasicConv2D(filters=256,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b3_conv2 = BasicConv2D(filters=288,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding="valid")
        self.b4_conv1 = BasicConv2D(filters=256,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv2 = BasicConv2D(filters=288,
                                    kernel_size=(3, 3),
                                    strides=1,
                                    padding="same")
        self.b4_conv3 = BasicConv2D(filters=320,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding="valid")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_maxpool(inputs)

        b2 = self.b2_conv1(inputs, training=training)
        b2 = self.b2_conv2(b2, training=training)

        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)

        b4 = self.b4_conv1(inputs, training=training)
        b4 = self.b4_conv2(b4, training=training)
        b4 = self.b4_conv3(b4, training=training)

        return tf.concat(values=[b1, b2, b3, b4], axis=-1)


def build_inception_resnet_a(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionResNetA())
    return block


def build_inception_resnet_b(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionResNetB())
    return block


def build_inception_resnet_c(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionResNetC())
    return block


class InceptionResNetV2(tf.keras.Model):
    def __init__(self):
        super(InceptionResNetV2, self).__init__()
        self.stem = Stem()
        self.inception_resnet_a = build_inception_resnet_a(5)
        self.reduction_a = ReductionA(k=256, l=256, m=384, n=384)
        self.inception_resnet_b = build_inception_resnet_b(10)
        self.reduction_b = ReductionB()
        self.inception_resnet_c = build_inception_resnet_c(5)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs, training=training)
        x = self.inception_resnet_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_resnet_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_resnet_c(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        x = self.fc(x)

        return x