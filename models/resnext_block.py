import tensorflow as tf


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 use_bias=True,
                 groups=1):
        super(GroupConv2D, self).__init__()
        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")
        self.groups = groups
        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv = tf.keras.layers.Conv2D(filters=self.group_out_num,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           dilation_rate=dilation_rate,
                                           use_bias=use_bias)
        self.feature_map_list = []

    def call(self, inputs, **kwargs):
        for i in range(self.groups):
            x_i = self.conv(inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            self.feature_map_list.append(x_i)
        out = tf.concat(self.feature_map_list, axis=-1)
        return out


class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self):
        super(ResNeXt_BottleNeck, self).__init__()
        pass

    def call(self, inputs, **kwargs):
        pass
