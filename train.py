import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

# model.py
from model import Model


# 将 .dat 文件读取并再存储
# def load(filename):
#     x = np.genfromtxt(filename, delimiter=",")
#     x = x.reshape((-1, 15, 15))
#     print("Loading",filename)
#     return x


# data = np.empty((0, 15, 15))
# for i in range(10):
#     x = load("x_" + str(i))
#     data = np.concatenate((data, x), axis=0)

# print(data.shape)
# np.savez_compressed("dataset/x_test.npz", data)

batch_size = 512


# 加载数据集
class GobangDataset(Dataset):
    def __init__(self, x_data, y_data):
        # 从文件加载
        self.x = np.load(x_data)["arr_0"]
        self.y = np.load(y_data)["arr_0"]
        if len(self.x) != len(self.y):
            print("Error loading dataset: len of X != len of Y. ")

        self.x = self.x.astype(np.float32).reshape(-1, 1, 15, 15)
        self.y = self.y.astype(np.float32).reshape(-1, 1, 15, 15)

        # 扩展数据集
        self.x = np.concatenate((self.x, np.rot90(self.x, k=2, axes=(2, 3))), axis=0)
        self.x = np.concatenate((self.x, np.rot90(self.x, k=1, axes=(2, 3))), axis=0)
        self.x = np.concatenate((self.x, np.flip(self.x)), axis=0)

        self.y = np.concatenate((self.y, np.rot90(self.y, k=2, axes=(2, 3))), axis=0)
        self.y = np.concatenate((self.y, np.rot90(self.y, k=1, axes=(2, 3))), axis=0)
        self.y = np.concatenate((self.y, np.flip(self.y)), axis=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


print("Loading dataset. . .")
train_dataset = GobangDataset("dataset/x_train.npz", "dataset/y_train.npz")
test_dataset = GobangDataset("dataset/x_test.npz", "dataset/y_test.npz")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print("Loaded dataset")
print("Train dataset Shape:", train_dataset.x.shape)
print("Test dataset Shape:", test_dataset.x.shape)

# 创建模型
print("Creating model. . .")
model = Model(15).cuda()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
# 打印可训练和不可训练参数量
print(f"Trainable parameters: {trainable_params}")
print(f"Non-trainable parameters: {non_trainable_params}")
# summary(model, (1, 1, 15, 15))


# 训练循环
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 获取数据集大小
    for batch, (x, y) in enumerate(dataloader):
        # 计算模型输出和误差
        pred = model(x.cuda())
        loss = loss_fn(pred.cuda(), y.cuda())

        # 反向传播（训练）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            # print(pred[0])
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 测试循环
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x.cuda())
            test_loss += loss_fn(pred, y.cuda()).item()
            correct += (
                (pred.argmax(1) == y.cuda().argmax(1)).type(torch.float).sum().item()
            )

    test_loss /= num_batches
    correct /= size * 15 * 15
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


# 设置超参数
learning_rate = 0.001
epochs = 10
loss_fn = nn.MSELoss()
# 设置优化器
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# 训练
print("Training model. . .")
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    model.train()
    train_loop(train_dataloader, model, loss_fn, optimizer)
    model.eval()
    test_loop(test_dataloader, model, loss_fn)
    # 保存模型
    print("Saving model. . .")
    torch.save(model, "model.pth")