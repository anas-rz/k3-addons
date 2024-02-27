# k3-addons: Additional multi-backend functionality for Keras 3.
![Logo](.assets/k-addons.png)

K3 Addons supercharge your multibackend Keras 3 workflow, giving access to various innovative machine learning techniques. While Keras 3 offers a rich set of APIs, not everything can be included in the core APIs due to less generic usage. K3 Addons bridges this gap, ensuring you're not limited by the core Keras 3 library. These add-ons might include various attention mechanisms for Text and Image Data, advanced optimizers, or specialized layers tailored for unique data types. With K3 Addons, you'll gain the flexibility to tackle emerging ML challenges and push the boundaries of what's possible with Keras 3.

# Installation
To Install K3 Addons simply run following command in your environment:

```bash
pip install k3-addons
```

# Includes:

Currently includes `layers`, `losses`, and `activations` API.

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

        Specific case of `keras.layers.GroupNormalization` since
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
