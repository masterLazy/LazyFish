# DeepFish-1.0

参数量：`78,497`。

采用 CNN 结构，在仓库中提供的数据集上训练。

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

