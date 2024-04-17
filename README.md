# LazyFish

五子棋 AI。

## 介绍
"LazyFish" 这个名字的灵感来自于 StockFish。LazyFish 包括两个版本：基于搜索和统计算法的 BasicFish，以及基于深度学习的 DeepFish。

LazyFish 尚未完善。

## 文件介绍

### 源文件

`gobang.hpp` 中将一些关于五子棋棋局的方法封装成一个类 `gobang::Board`，并定义了下棋模块接口 `gobang::Fish`。

`basic_fish.hpp` 实现了 BasicFish。

`deep_fish.hpp` 实现了一个接口 `DeepFish`，允许用户加载和使用 DeepFish 模型（必须是序列化后的 `.pt` 文件）。

`dataset_maker.cpp` 是一个用于制造两个 BasicFish 之间互相对弈的数据集的程序。`dataset/` 下存储了用它制造的数据集（`.npz` 格式）

`gui_demo.cpp` 是一个图形化下棋程序，要运行它，你需要安装 `mLib`（见我的仓库：[mLib](https://github.com/masterLazy/mLib)）。

`train.py` 是一个训练程序，包括读取 `dataset_maker.cpp` 制造的数据并存储为 `.npz`、扩展数据集以及训练过程。

### 模型文件

`model/` 下存储了 DeepFish 的模型文件（`.pth`）、序列化模型文件（`.pt`）和定义它们的 Python 源代码。

## DeepFish 输入约定

DeepFish 的输入 `shape` 是：`(1,15,15)`

`0` = 空位

`0.5` = 己方

`1.0` = 敌方
