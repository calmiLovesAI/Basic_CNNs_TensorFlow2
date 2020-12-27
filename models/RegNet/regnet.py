import numpy as np
import models.RegNet.blocks as bk

from models.RegNet.anynet import AnyNet
from anynet_cfg import RegNetCfg
from configuration import NUM_CLASSES


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


class RegNet(AnyNet):
    @staticmethod
    def get_params():
        w_a, w_0, w_m, d = RegNetCfg.wa, RegNetCfg.w0, RegNetCfg.wm, RegNetCfg.depth
        ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
        ss = [RegNetCfg.stride for _ in ws]
        bs = [RegNetCfg.bot_mul for _ in ws]
        gs = [RegNetCfg.group_w for _ in ws]
        ws, bs, gs = bk.adjust_block_compatibility(ws, bs, gs)

        return {
            "stem_type": RegNetCfg.stem_type,
            "stem_w": RegNetCfg.stem_w,
            "block_type": RegNetCfg.block_type,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "se_r": RegNetCfg.se_r if RegNetCfg.se_on else 0,
            "num_classes": NUM_CLASSES,
        }

    def __init__(self):
        params = RegNet.get_params()
        super(RegNet, self).__init__(params)
