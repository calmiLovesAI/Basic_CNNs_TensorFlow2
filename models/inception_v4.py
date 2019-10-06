import tensorflow as tf
from models.inception_modules import Stem, InceptionBlockA, InceptionBlockB, \
    InceptionBlockC, ReductionA, ReductionB
from configuration import NUM_CLASSES


def build_inception_block_a(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockA())
    return block


def build_inception_block_b(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockB())
    return block


def build_inception_block_c(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockC())
    return block


class InceptionV4(tf.keras.Model):
    def __init__(self):
        super(InceptionV4, self).__init__()
        self.stem = Stem()
        self.inception_a = build_inception_block_a(4)
        self.reduction_a = ReductionA(k=192, l=224, m=256, n=384)
        self.inception_b = build_inception_block_b(7)
        self.reduction_b = ReductionB()
        self.inception_c = build_inception_block_c(3)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=True, mask=None):
        x = self.stem(inputs, training=training)
        x = self.inception_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        x = self.fc(x)

        return x
