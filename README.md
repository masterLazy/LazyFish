# LazyFish

五子棋 AI。

## 介绍
"LazyFish" 这个名字的灵感来自于 StockFish。LazyFish 包括两个版本：基于搜索和统计算法的 BasicFish，以及基于深度学习的 DeepFish。

LazyFish 持续开发中。

## 文件介绍

### 源文件

`gobang.hpp` 中将一些关于五子棋棋局的方法封装成一个类 `gobang::Board`，并定义了下棋模块接口 `gobang::Fish`。

`basic_fish.hpp` 实现了 BasicFish。

`deep_fish.hpp` 实现了一个接口 `DeepFish`，允许用户加载和使用 DeepFish 模型（必须是序列化后的 `.pt` 文件）。要运行这个源文件，你需要先安装 LibTorch（PyTorch 的 C++ 接口）。

`gobang.pyd` 是 `gobang::Board` 的 Python 接口。

程序示例见 `demo/`。

### 模型文件

`model/` 下存储了 DeepFish 的模型文件（`.pth`）、序列化模型文件（`.pt`）和定义它们的 Python 源代码。

### 数据集

见 `dataset/` 。

## DeepFish 输入约定

DeepFish 的输入 `shape` 是：`(1,1,15,15)`

`0` = 空位

`0.5` = 己方

`1.0` = 敌方