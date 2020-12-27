

class AnyNetCfg:
    stem_type = "simple_stem_in"
    stem_w = 32
    block_type = "res_bottleneck_block"
    depths = []
    widths = []
    strides = []
    bot_muls = []
    group_ws = []
    se_on = True
    se_r = 0.25


class RegNetCfg:
    stem_type = "simple_stem_in"
    stem_w = 32
    block_type = "res_bottleneck_block"
    stride = 2
    se_on = True
    se_r = 0.25
    depth = 10
    w0 = 32
    wa = 5.0
    wm = 2.5
    group_w = 16
    bot_mul = 1.0
