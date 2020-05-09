# Basic_CNNs_TensorFlow2
A tensorflow2 implementation of some basic CNNs.

## Networks included:
+ MobileNet_V1
+ MobileNet_V2
+ [MobileNet_V3](https://github.com/calmisential/MobileNetV3_TensorFlow2)
+ [EfficientNet](https://github.com/calmisential/EfficientNet_TensorFlow2)
+ [ResNeXt](https://github.com/calmisential/ResNeXt_TensorFlow2)
+ [InceptionV4, InceptionResNetV1, InceptionResNetV2](https://github.com/calmisential/InceptionV4_TensorFlow2)
+ SE_ResNet_50, SE_ResNet_101, SE_ResNet_152, SE_ResNeXt_50, SE_ResNeXt_101
+ SqueezeNet
+ [DenseNet](https://github.com/calmisential/DenseNet_TensorFlow2)
+ ShuffleNetV2
+ [ResNet](https://github.com/calmisential/TensorFlow2.0_ResNet)

## Other networks
For AlexNet and VGG, see : https://github.com/calmisential/TensorFlow2.0_Image_Classification<br/>
For InceptionV3, see : https://github.com/calmisential/TensorFlow2.0_InceptionV3<br/>
For ResNet, see : https://github.com/calmisential/TensorFlow2.0_ResNet

## Train
1. Requirements:
+ Python >= 3.6
+ Tensorflow >= 2.2.0rc3
2. To train the network on your own dataset, you can put the dataset under the folder **original dataset**, and the directory should look like this:
```
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```
3. Run the script **split_dataset.py** to split the raw dataset into train set, valid set and test set. The dataset directory will be like this:
 ```
|——dataset
   |——train
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |——valid
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |—-test
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
```
4. Run **to_tfrecord.py** to generate tfrecord files.
5. Change the corresponding parameters in **config.py**.
6. Run **train.py** to start training.<br/>
If you want to train the *EfficientNet*, you should change the IMAGE_HEIGHT and IMAGE_WIDTH to *resolution* in the params, and then run **train.py** to start training.
## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.

## Different input image sizes for different neural networks
<table>
     <tr align="center">
          <th>Type</th>
          <th>Neural Network</th>
          <th>Input Image Size (height * width)</th>
     </tr>
     <tr align="center">
          <td rowspan="3">MobileNet</td>
          <td>MobileNet_V1</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>MobileNet_V2</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>MobileNet_V3</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>EfficientNet</td>
          <td>EfficientNet(B0~B7)</td>
          <td>/</td>
     </tr>
     <tr align="center">
          <td rowspan="2">ResNeXt</td>
          <td>ResNeXt50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNeXt101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="2">SEResNeXt</td>
          <td>SEResNeXt50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SEResNeXt101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="3">Inception</td>
          <td>InceptionV4</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td>Inception_ResNet_V1</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td>Inception_ResNet_V2</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td rowspan="3">SE_ResNet</td>
          <td>SE_ResNet_50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SE_ResNet_101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SE_ResNet_152</td>
          <td>(224 * 224)</td>
     </tr>
     </tr align="center">
          <td>SqueezeNet</td>
          <td align="center">SqueezeNet</td>
          <td align="center">(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="4">DenseNet</td>
          <td>DenseNet_121</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_169</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_201</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_269</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ShuffleNetV2</td>
          <td>ShuffleNetV2</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="5">ResNet</td>
          <td>ResNet_18</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_34</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_152</td>
          <td>(224 * 224)</td>
     </tr>
</table>

## References
1. MobileNet_V1: [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
2. MobileNet_V2: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
3. MobileNet_V3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
4. EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
5. The official code of EfficientNet: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
6. ResNeXt: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
7. Inception_V4/Inception_ResNet_V1/Inception_ResNet_V2: [Inception-v4,  Inception-ResNet and the Impact of Residual Connectionson Learning](https://arxiv.org/abs/1602.07261)
8. The official implementation of Inception_V4: https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py
9. The official implementation of Inception_ResNet_V2: https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
10. SENet: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
11. SqueezeNet: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
12. DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
13. https://zhuanlan.zhihu.com/p/37189203
14. ShuffleNetV2: [ShuffleNet V2: Practical Guidelines for Eﬃcient CNN Architecture Design
](https://arxiv.org/abs/1807.11164)
15. https://zhuanlan.zhihu.com/p/48261931
16. ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)