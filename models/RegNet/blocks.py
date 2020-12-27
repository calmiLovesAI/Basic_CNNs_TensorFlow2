import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class SE(tf.keras.layers.Layer):
    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1))

        self.c1 = tf.keras.layers.Conv2D(filters=w_se, kernel_size=1, padding="same")
        self.c2 = tf.keras.layers.Conv2D(filters=w_in, kernel_size=1, padding="same")

    def call(self, inputs, **kwargs):
        x = self.avg_pool(inputs)
        x = tf.nn.relu(self.c1(x))
        x = tf.nn.sigmoid(self.c2(x))
        return x


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, b) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs