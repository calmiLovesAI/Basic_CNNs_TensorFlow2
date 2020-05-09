import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         data_format=data_format,
                                                         dilation_rate=dilation_rate,
                                                         activation=activations.get(activation),
                                                         use_bias=use_bias,
                                                         kernel_initializer=initializers.get(kernel_initializer),
                                                         bias_initializer=initializers.get(bias_initializer),
                                                         kernel_regularizer=regularizers.get(kernel_regularizer),
                                                         bias_regularizer=regularizers.get(bias_regularizer),
                                                         activity_regularizer=regularizers.get(activity_regularizer),
                                                         kernel_constraint=constraints.get(kernel_constraint),
                                                         bias_constraint=constraints.get(bias_constraint),
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out

    def get_config(self):
        config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "groups": self.groups,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint)
        }
        base_config = super(GroupConv2D, self).get_config()
        return {**base_config, **config}


class GroupConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        super(GroupConv2DTranspose, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2DTranspose(filters=self.group_out_num,
                                                                  kernel_size=kernel_size,
                                                                  strides=strides,
                                                                  padding=padding,
                                                                  output_padding=output_padding,
                                                                  data_format=data_format,
                                                                  dilation_rate=dilation_rate,
                                                                  activation=activations.get(activation),
                                                                  use_bias=use_bias,
                                                                  kernel_initializer=initializers.get(kernel_initializer),
                                                                  bias_initializer=initializers.get(bias_initializer),
                                                                  kernel_regularizer=regularizers.get(kernel_regularizer),
                                                                  bias_regularizer=regularizers.get(bias_regularizer),
                                                                  activity_regularizer=regularizers.get(activity_regularizer),
                                                                  kernel_constraint=constraints.get(kernel_constraint),
                                                                  bias_constraint=constraints.get(bias_constraint),
                                                                  **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out

    def get_config(self):
        config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "output_padding": self.output_padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "groups": self.groups,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint)
        }
        base_config = super(GroupConv2DTranspose, self).get_config()
        return {**base_config, **config}
