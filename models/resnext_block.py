import tensorflow as tf


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 use_bias=True,
                 groups=1):
        super(GroupConv2D, self).__init__()
        self.groups = groups
        self.group_feature_map_num = filters // groups
        self.conv = tf.keras.layers.Conv2D(filters=self.group_feature_map_num,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           dilation_rate=dilation_rate,
                                           use_bias=use_bias)
        self.feature_map_list = []

    def call(self, inputs, **kwargs):
        for i in range(self.groups):
            x_i = self.conv(inputs[:, :, :, i*self.group_feature_map_num: (i + 1) * self.group_feature_map_num])
            self.feature_map_list.append(x_i)
        out = tf.concat(self.feature_map_list, axis=-1)
        return out


class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self):
        super(ResNeXt_BottleNeck, self).__init__()
        pass

    def call(self, inputs, **kwargs):
        pass