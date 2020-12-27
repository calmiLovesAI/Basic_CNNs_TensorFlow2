import tensorflow as tf
import tensorflow_addons as tfa

from models.RegNet.blocks import SE
from anynet_cfg import AnyNetCfg
from configuration import NUM_CLASSES
from models.group_convolution import get_group_conv


def get_stem_fun(stem_type):
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "simple_stem_in": SimpleStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(AnyHead, self).__init__()
        self.avg_pool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=num_classes, use_bias=True)

    def call(self, inputs, **kwargs):
        x = self.avg_pool(inputs)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class VanillaBlock(tf.keras.layers.Layer):
    def __init__(self, w_out, stride):
        super(VanillaBlock, self).__init__()
        self.a = tf.keras.layers.Conv2D(filters=w_out, kernel_size=3, strides=stride, padding="same")
        self.a_bn = tf.keras.layers.BatchNormalization()
        self.b = tf.keras.layers.Conv2D(filters=w_out, kernel_size=3, padding="same")
        self.b_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.a(inputs)
        x = self.a_bn(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        x = tf.keras.activations.relu(x)
        return x


class BasicTransform(tf.keras.layers.Layer):
    def __init__(self, w_out, stride):
        super(BasicTransform, self).__init__()
        self.a = tf.keras.layers.Conv2D(filters=w_out, kernel_size=3, strides=stride, padding="same")
        self.a_bn = tf.keras.layers.BatchNormalization()
        self.b = tf.keras.layers.Conv2D(filters=w_out, kernel_size=3, padding="same")
        self.b_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.a(inputs)
        x = self.a_bn(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        return x


class ResBasicBlock(tf.keras.layers.Layer):
    def __init__(self, w_in, w_out, stride):
        super(ResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = tf.keras.layers.Conv2D(filters=w_out, kernel_size=1, strides=stride, padding="same")
            self.bn = tf.keras.layers.BatchNormalization()
        self.f = BasicTransform(w_out, stride)

    def call(self, inputs, training=None, **kwargs):
        if self.proj:
            x = self.proj(inputs)
            x = self.bn(x, training=training)
        else:
            x = inputs
        return tf.nn.relu(x + self.f(inputs, training=training))


class BottleneckTransform(tf.keras.layers.Layer):
    def __init__(self, w_in, w_out, stride, params):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]

        self.a = tf.keras.layers.Conv2D(filters=w_b, kernel_size=1, strides=1, padding="same")
        self.a_bn = tf.keras.layers.BatchNormalization()
        # self.b = tf.keras.layers.Conv2D(filters=w_b, kernel_size=3, strides=stride, padding="same", groups=groups)
        self.b = get_group_conv(in_channels=w_b, out_channels=w_b, kernel_size=3, strides=stride, padding="same", groups=groups)
        self.b_bn = tf.keras.layers.BatchNormalization()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = tf.keras.layers.Conv2D(filters=w_out, kernel_size=1, strides=1, padding="same")
        self.c_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.a(inputs)
        x = self.a_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.se(x)
        x = self.c(x)
        x = self.c_bn(x, training=training)
        return x


class ResBottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = tf.keras.layers.Conv2D(filters=w_out, kernel_size=1, strides=stride, padding="same")
            self.bn = tf.keras.layers.BatchNormalization()
        self.f = BottleneckTransform(w_in, w_out, stride, params)

    def call(self, inputs, training=None, **kwargs):
        if self.proj:
            x = self.proj(inputs)
            x = self.bn(x, training=training)
        else:
            x = inputs
        return tf.nn.relu(x + self.f(inputs, training=training))


class ResStemCifar(tf.keras.layers.Layer):
    def __init__(self, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=w_out, kernel_size=3, strides=1, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class ResStem(tf.keras.layers.Layer):
    def __init__(self, w_out):
        super(ResStem, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=w_out, kernel_size=7, strides=2, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()

        self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)
        return x


class SimpleStem(tf.keras.layers.Layer):
    def __init__(self, w_out):
        super(SimpleStem, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=w_out, kernel_size=3, strides=2, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class AnyStage(tf.keras.layers.Layer):
    def __init__(self, w_in, w_out, stride, d, block_fun, params):
        super(AnyStage, self).__init__()
        self.layers = list()
        for _ in range(d):
            self.block = block_fun(w_in, w_out, stride, params)
            self.layers.append(self.block)
            stride, w_in = 1, w_out

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for l in self.layers:
            x = l(x, training=training)
        return x


class AnyNet(tf.keras.Model):
    def __init__(self, params=None):
        super(AnyNet, self).__init__()
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]

        self.stages = list()

        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            stage = AnyStage(prev_w, w, s, d, block_fun, params)
            self.stages.append(stage)
            prev_w = w
        self.head = AnyHead(p["num_classes"])

    @staticmethod
    def get_params():
        nones = [None for _ in AnyNetCfg.depths]
        return {
            "stem_type": AnyNetCfg.stem_type,
            "stem_w": AnyNetCfg.stem_w,
            "block_type": AnyNetCfg.block_type,
            "depths": AnyNetCfg.depths,
            "widths": AnyNetCfg.widths,
            "strides": AnyNetCfg.strides,
            "bot_muls": AnyNetCfg.bot_muls if AnyNetCfg.bot_muls else nones,
            "group_ws": AnyNetCfg.group_ws if AnyNetCfg.group_ws else nones,
            "se_r": AnyNetCfg.se_r if AnyNetCfg.se_on else 0,
            "num_classes": NUM_CLASSES,
        }

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs, training=training)
        for s in self.stages:
            x = s(x, training=training)
        x = self.head(x)
        return x