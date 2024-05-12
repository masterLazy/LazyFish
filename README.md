# LazyFish

五子棋 AI。

## 介绍
"LazyFish" 这个名字的灵感来自于 StockFish。LazyFish 包括若干版本：

1. 我开发的“经典”算法——BasicFish。
2. 完全基于深度学习的 DeepFish。
3. 基于搜索算法的 [unnamed]Fish。

LazyFish 持续开发中。

## 文件介绍

### 源文件

`gobang.hpp` 中将一些关于五子棋棋局的方法封装成一个类 `gobang::Board`，并定义了下棋模块接口 `gobang::Fish`。

`basic_fish.hpp` 实现了 BasicFish。

`deep_fish.hpp` 实现了一个接口 `DeepFish`，允许用户加载和使用 DeepFish 模型（必须是序列化后的 `.pt` 文件）。要运行这个源文件，你需要先安装 LibTorch（PyTorch 的 C++ 接口）。

`gobang.pyd` 是 `gobang::Board` 的 Python 接口。

程序示例见 `demo/`。

### 数据集

见 `dataset/` 。

## BasicFish

我编写的最早的五子棋算法，保留了“原味”，但是做了一些优化。具体介绍请见：[BasicFish](BASICFISH.md)。

## DeepFish

有关 DeepFish 的具体介绍请见`model/`。此目录下存储了 DeepFish 的模型文件（`.pth`）、序列化模型文件（`.pt`）和定义它们的 Python 源代码（还有一份 `README.md`）。