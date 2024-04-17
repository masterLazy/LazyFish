#pragma once
/*****************************************************************************
* deep_fish.hpp
*****************************************************************************/

#include <torch/torch.h>
#include <torch/script.h>
#include "gobang.hpp"

#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "caffe2_nvrtc.lib")
#pragma comment(lib, "clog.lib")
#pragma comment(lib, "cpuinfo.lib")
#pragma comment(lib, "dnnl.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "fmt.lib")
#pragma comment(lib, "kineto.lib")
#pragma comment(lib, "libprotobuf-lite.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libprotoc.lib")
#pragma comment(lib, "pthreadpool.lib")
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "XNNPACK.lib")

namespace gobang
{
	class DeepFish :Fish
	{
	private:
		size_t num_params;
	public:
		torch::jit::script::Module model;
		explicit DeepFish(std::string model_path)
		{
			//加载模型
			try
			{
				model = torch::jit::load("D:/model.pt");
			}
			catch (const c10::Error& e)
			{
				std::cerr << "Error loading DeepFish model." << std::endl;
			}
			auto parameters = model.parameters();
			// 计算参数总数
			for (const auto& param : parameters)
			{
				num_params += param.numel();
			}
			if (torch::cuda::is_available())
			{
				model.to(c10::Device::Type::CUDA);
			}
		}

		std::array<int, 2> play(Board bd, int self, bool fbd = true)
		{
			//准备输入
			auto map = torch::zeros({ 1,15,15 });
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd[x][y] == self)map[0][x][y] = 0.5;
					else if (bd[x][y] == Invert(self))map[0][x][y] = 1;
				}
			}

			//模型预测
			auto pred = model({ map }).toTensor();
			std::cout << "Shape of pred: " << pred.sizes() << std::endl;

			//获得输出
			int _x = -1, _y = -1;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd.is_able(x, y, self, fbd))
					{
						//if (_x == -1 || pred[0][x][y].item() > pred[0][_x][_y].item())
						{
							_x = x;
							_y = y;
						}
						//else if (m[x][y] == m[_x][_y] && rand() % 15 * 15 == 0)
						{
							_x = x;
							_y = y;
						}
					}
				}
			}
			return { _x,_y };
		}
	};
}