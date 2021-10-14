from . import mobilenet_v1, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    efficientnet, resnext, inception_v4, inception_resnet_v1, inception_resnet_v2, \
    se_resnet, squeezenet, densenet, shufflenet_v2, resnet, se_resnext
from .RegNet import regnet

model_list = [
    mobilenet_v1.MobileNetV1(), mobilenet_v2.MobileNetV2(), mobilenet_v3_large.MobileNetV3Large(),
    mobilenet_v3_small.MobileNetV3Small(),
    efficientnet.efficient_net_b0(), efficientnet.efficient_net_b1(),
    efficientnet.efficient_net_b2(), efficientnet.efficient_net_b3(), efficientnet.efficient_net_b4(),
    efficientnet.efficient_net_b5(), efficientnet.efficient_net_b6(), efficientnet.efficient_net_b7(),
    resnext.ResNeXt50(), resnext.ResNeXt101(),
    inception_v4.InceptionV4(),
    inception_resnet_v1.InceptionResNetV1(),
    inception_resnet_v2.InceptionResNetV2(),
    se_resnet.se_resnet_50(), se_resnet.se_resnet_101(), se_resnet.se_resnet_152(),
    se_resnext.SEResNeXt50(), se_resnext.SEResNeXt101(),
    resnet.resnet_18(), resnet.resnet_34(), resnet.resnet_50(), resnet.resnet_101(), resnet.resnet_152(),
    shufflenet_v2.shufflenet_0_5x(), shufflenet_v2.shufflenet_1_0x(), shufflenet_v2.shufflenet_1_5x(),
    shufflenet_v2.shufflenet_2_0x(),
    regnet.RegNet()
]


def get_model2idx_dict():
    return dict((v, k) for k, v in enumerate(model_list))


def get_model(idx):
    return model_list[idx]
