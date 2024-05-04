from torch.utils.data import Dataset

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