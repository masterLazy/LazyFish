# DeepFish-2.0

参数量：`4,938,225`。

采用 CNN 结构，相比于 DeepFish-1.0，主要是加入了一个 FNN，并且降低了中间层的通道数。

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [1, 1, 15, 15]            --
├─Sequential: 1-1                        [1, 16, 7, 7]             --
│    └─Conv2d: 2-1                       [1, 4, 11, 11]            104
│    └─LeakyReLU: 2-2                    [1, 4, 11, 11]            --
│    └─LayerNorm: 2-3                    [1, 4, 11, 11]            968
│    └─Conv2d: 2-4                       [1, 8, 9, 9]              296
│    └─LeakyReLU: 2-5                    [1, 8, 9, 9]              --
│    └─LayerNorm: 2-6                    [1, 8, 9, 9]              1,296
│    └─Conv2d: 2-7                       [1, 16, 7, 7]             1,168
│    └─LeakyReLU: 2-8                    [1, 16, 7, 7]             --
│    └─LayerNorm: 2-9                    [1, 16, 7, 7]             1,568
├─Sequential: 1-2                        [1, 784]                  --
│    └─Linear: 2-10                      [1, 3136]                 2,461,760
│    └─LeakyReLU: 2-11                   [1, 3136]                 --
│    └─LayerNorm: 2-12                   [1, 3136]                 6,272
│    └─Linear: 2-13                      [1, 784]                  2,459,408
│    └─LeakyReLU: 2-14                   [1, 784]                  --
│    └─LayerNorm: 2-15                   [1, 784]                  1,568
├─Sequential: 1-3                        [1, 1, 15, 15]            --
│    └─ConvTranspose2d: 2-16             [1, 8, 9, 9]              1,160
│    └─LeakyReLU: 2-17                   [1, 8, 9, 9]              --
│    └─LayerNorm: 2-18                   [1, 8, 9, 9]              1,296
│    └─ConvTranspose2d: 2-19             [1, 4, 11, 11]            292
│    └─LeakyReLU: 2-20                   [1, 4, 11, 11]            --
│    └─LayerNorm: 2-21                   [1, 4, 11, 11]            968
│    └─ConvTranspose2d: 2-22             [1, 1, 15, 15]            101
│    └─LeakyReLU: 2-23                   [1, 1, 15, 15]            --
├─Softmax: 1-4                           [1, 225]                  --
==========================================================================================
Total params: 4,938,225
Trainable params: 4,938,225
Non-trainable params: 0
Total mult-adds (M): 5.18
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.11
Params size (MB): 19.75
Estimated Total Size (MB): 19.87
==========================================================================================
```

### 训练超参数

```python
batch_size = 512
learning_rate = 0.0001
epochs = 2
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
