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
		c10::Device::Type device;
	public:
		torch::jit::script::Module model;
		DeepFish() {}
		explicit DeepFish(std::string model_path)
		{
			load(model_path);
		}

		bool load(std::string model_path)
		{
			//加载模型
			try
			{
				model = torch::jit::load(model_path.c_str());
			}
			catch (const c10::Error& e)
			{
				std::cerr << "[DeepFish] Error loading model: " << e.what() << std::endl;
				return false;
			}
			auto parameters = model.parameters();
			//计算参数总数
			num_params = 0;
			for (const auto& param : parameters)
			{
				num_params += param.numel();
			}
			if (torch::cuda::is_available())
			{
				device = c10::Device::Type::CUDA;
				model.to(device);
			}
			else
			{
				device = c10::Device::Type::CPU;
			}
			return true;
		}

		std::array<int, 2> play(Board bd, int self, bool fbd = true)
		{
			if (num_params == 0)
			{
				std::cerr << "[DeepFish] Error predicting: Model not loaded.";
				return { -1,-1 };
			}
			//准备输入
			auto map = torch::zeros({ 15,15 });
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd[x][y] == self)map[x][y] = 0.5;
					else if (bd[x][y] == Invert(self))map[x][y] = 1;
				}
			}
			map = map.reshape({ 1,1,15,15 }).to(device);

			//模型预测
			c10::IValue pred;
			try
			{
				pred = model({ map });
			}
			catch (const std::exception& e)
			{
				std::cerr << "[DeepFish] Error predicting: " << e.what() << std::endl;
				return { -1,-1 };
			}
			auto pred_ = pred.toTensor();

			//获得输出
			float max_ = 0;
			int _x = -1, _y = -1;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd.is_able(x, y, self, fbd))
					{
						auto n = pred_[0][0][x][y].item().toFloat();
						if (_x == -1 || n > max_)
						{
							_x = x;
							_y = y;
							max_ = n;
						}
						else if (n == max_ && rand() % 15 * 15 == 0)
						{
							_x = x;
							_y = y;
							max_ = n;
						}
					}
				}
			}
			return { _x,_y };
		}
	};
}