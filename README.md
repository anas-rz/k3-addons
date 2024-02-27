# k3-addons: Additional multi-backend functionality for Keras 3.
![Logo](.assets/k-addons.png)

# Installation

```bash
pip install k3-addons
```

# Includes:
- ## Layers
- ### Pooling:
    - `k3_addons.layers.AdaptiveAveragePooling1D`
        Multibackend Implementation of `torch.nn.AdaptiveAvgPool1d`. The results are close to PyTorch.  
    - `k3_addons.layers.AdaptiveMaxPooling1D`
        Multibackend Implementation of `torch.nn.AdaptiveMaxPool1d`. The results are close to PyTorch.  
    - `k3_addons.layers.AdaptiveAveragePooling2D`
        Multibackend Implementation of `torch.nn.AdaptiveAvgPool2d`. The results are close to PyTorch.  
    - `k3_addons.layers.AdaptiveMaxPooling2D`
        Multibackend Implementation of `torch.nn.AdaptiveMaxPool2d`. The results are close to PyTorch.  
    - `k3_addons.layers.Maxout`
        Multibackend port of `tensorflow_addons.layers.Maxout`. [Paper](https://arxiv.org/abs/1302.4389)

- #### Normalization
    - `k3_addons.layers.InstanceNormalization`
        specific case of `keras.layers.GroupNormalization` since
        it normalizes all features of one channel. The Groupsize is equal to the
        channel size. 
- #### Attention:
    - `k3_addons.layers.DoubleAttention`
    [Paper](https://arxiv.org/pdf/1810.11579.pdf)
    - `k3_addons.layers.AFTFull`
    [An Attention Free Transformer](https://arxiv.org/pdf/2105.14103v1.pdf)
    - `k3_addons.layers.ChannelAttention2D`
    - `k3_addons.layers.SpatialAttention2D`
    - `k3_addons.layers.ECAAttention`
    [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/pdf/1910.03151.pdf)
    - `k3_addons.layers.ExternalAttention`
    [Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks](https://arxiv.org/abs/2105.02358)
    - `k3_addons.layers.ResidualAttention`
    [Residual Attention: A Simple but Effective Method for Multi-Label Recognition](https://arxiv.org/abs/2108.02456)
    - `k3_addons.layers.MobileViTAttention`
    [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
    - `k3_addons.layers.BAMBlock`
    - `k3_addons.layers.CBAM`
    - `k3_addons.layers.MobileViTv2Attention`
    [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)
    - `k3_addons.layers.ParNetAttention`
    [Non-deep Networks](https://arxiv.org/abs/2110.07641)
    - `k3_addons.layers.SimAM`

- ## Losses
    - `k3_addons.losses.ContrastiveLoss`
    - `k3_addons.losses.GIoULoss`
    - `k3_addons.losses.PinballLoss`
    - `k3_addons.losses.SigmoidFocalCrossEntropy`
    - `k3_addons.losses.WeightedKappaLoss`
    - `k3_addons.losses.pairwise_distance`
    - `k3_addons.losses.pinball_loss`

- ## Activations

    -  `k3_addons.activations.hardshrink`
    - `k3_addons.activations.lisht`
    - `k3_addons.activations.mish`
    - `k3_addons.activations.snake`
    - `k3_addons.activations.tanhshrink`
