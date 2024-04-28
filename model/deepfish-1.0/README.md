# DeepFish-1.0

参数量：`78,497`。

采用 CNN 结构，在仓库中提供的数据集上训练。

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [1, 1, 15, 15]            --
├─Sequential: 1-1                        [1, 1, 15, 15]            --
│    └─Conv2d: 2-1                       [1, 8, 13, 13]            80
│    └─LeakyReLU: 2-2                    [1, 8, 13, 13]            --
│    └─LayerNorm: 2-3                    [1, 8, 13, 13]            2,704
│    └─Conv2d: 2-4                       [1, 16, 11, 11]           1,168
│    └─LeakyReLU: 2-5                    [1, 16, 11, 11]           --
│    └─LayerNorm: 2-6                    [1, 16, 11, 11]           3,872
│    └─Conv2d: 2-7                       [1, 32, 9, 9]             4,640
│    └─LeakyReLU: 2-8                    [1, 32, 9, 9]             --
│    └─LayerNorm: 2-9                    [1, 32, 9, 9]             5,184
│    └─Conv2d: 2-10                      [1, 64, 7, 7]             18,496
│    └─LeakyReLU: 2-11                   [1, 64, 7, 7]             --
│    └─LayerNorm: 2-12                   [1, 64, 7, 7]             6,272
│    └─ConvTranspose2d: 2-13             [1, 32, 9, 9]             18,464
│    └─LeakyReLU: 2-14                   [1, 32, 9, 9]             --
│    └─LayerNorm: 2-15                   [1, 32, 9, 9]             5,184
│    └─ConvTranspose2d: 2-16             [1, 16, 11, 11]           4,624
│    └─LeakyReLU: 2-17                   [1, 16, 11, 11]           --
│    └─LayerNorm: 2-18                   [1, 16, 11, 11]           3,872
│    └─ConvTranspose2d: 2-19             [1, 8, 13, 13]            1,160
│    └─LeakyReLU: 2-20                   [1, 8, 13, 13]            --
│    └─LayerNorm: 2-21                   [1, 8, 13, 13]            2,704
│    └─ConvTranspose2d: 2-22             [1, 1, 15, 15]            73
│    └─LeakyReLU: 2-23                   [1, 1, 15, 15]            --
├─Softmax: 1-2                           [1, 225]                  --
==========================================================================================
Total params: 78,497
Trainable params: 78,497
Non-trainable params: 0
Total mult-adds (M): 3.73
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.24
Params size (MB): 0.31
Estimated Total Size (MB): 0.56
==========================================================================================
```

```
Test Error:
 Accuracy: 100.0%, Avg loss: 0.003641
```

### 训练超参数

```python
batch_size = 512
learning_rate = 0.0001
epochs = 10
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

### 评价

具有一定的棋力，验证了模型的可行性，但是还不如 BasicFish，有点傻。
