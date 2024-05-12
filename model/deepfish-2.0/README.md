# DeepFish-2.0

参数量：`76,873`。

采用 U-Net 结构，在 `dataset/dataset-1.1.zip` 上训练。

```
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
Model                                              [1, 1, 15, 15]            --
├─ConvBlock: 1-1                                   [1, 8, 15, 15]            --
│    └─Sequential: 2-1                             [1, 8, 15, 15]            --
│    │    └─Conv2d: 3-1                            [1, 8, 15, 15]            208
│    │    └─BatchNorm2d: 3-2                       [1, 8, 15, 15]            16
│    │    └─ReLU: 3-3                              [1, 8, 15, 15]            --
│    │    └─Conv2d: 3-4                            [1, 8, 15, 15]            1,608
│    │    └─BatchNorm2d: 3-5                       [1, 8, 15, 15]            16
│    │    └─ReLU: 3-6                              [1, 8, 15, 15]            --
├─ModuleList: 1-2                                  --                        --
│    └─DownSample: 2-2                             [1, 16, 7, 7]             --
│    │    └─Sequential: 3-7                        [1, 16, 7, 7]             9,696
│    └─DownSample: 2-3                             [1, 32, 3, 3]             --
│    │    └─Sequential: 3-8                        [1, 32, 3, 3]             38,592
├─ModuleList: 1-3                                  --                        --
│    └─UpSample: 2-4                               [1, 16, 7, 7]             --
│    │    └─ConvTranspose2d: 3-9                   [1, 16, 7, 7]             2,064
│    │    └─ConvBlock: 3-10                        [1, 16, 7, 7]             19,296
│    └─UpSample: 2-5                               [1, 8, 15, 15]            --
│    │    └─ConvTranspose2d: 3-11                  [1, 8, 15, 15]            520
│    │    └─ConvBlock: 3-12                        [1, 8, 15, 15]            4,848
├─Conv2d: 1-4                                      [1, 1, 15, 15]            9
├─Softmax: 1-5                                     [1, 225]                  --
====================================================================================================
Total params: 76,873
Trainable params: 76,873
Non-trainable params: 0
Total mult-adds (M): 3.47
====================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.20
Params size (MB): 0.31
Estimated Total Size (MB): 0.51
====================================================================================================
```

### 训练超参数

```python
batch_size = 512
learning_rate = 0.0001
epochs = 4
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
