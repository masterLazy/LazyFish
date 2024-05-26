#pragma once
/*****************************************************************************
* deep_ee.hpp
* 
* Deep Estimation Engine
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
	class DeepEE
	{
	private:
		size_t num_params = 0;
		c10::DeviceType device;
	public:
		torch::jit::script::Module model;
		DeepEE() {}
		explicit DeepEE(std::string model_path, bool warm_up = true)
		{
			load(model_path, warm_up);
		}

		bool load(std::string model_path, bool warm_up = true)
		{
			//����ģ��
			try
			{
				model = torch::jit::load(model_path.c_str());
			}
			catch (const c10::Error& e)
			{
				std::cerr << "[DeepEE] Error loading model: " << e.what() << std::endl;
				return false;
			}
			auto parameters = model.parameters();
			//�����������
			num_params = 0;
			for (const auto& param : parameters)
			{
				num_params += param.numel();
			}
			if (torch::cuda::is_available())
			{
				device = c10::DeviceType::CUDA;
				model.to(device);
			}
			else
			{
				device = c10::DeviceType::CPU;
			}
			//��һ��ʹ��ģ��Ԥ����е�������������Ԥ��һ��
			if (warm_up)
			{
				model({ torch::zeros({1,1,15,15}).to(device) });
			}
			return true;
		}


		//[0,1]��Ԥ�����
		float predict(Board bd, int self)
		{
			if (num_params == 0)
			{
				std::cerr << "[DeepEE] Error predicting: Model not loaded.";
				return NAN;
			}
			//׼������
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

			//ģ��Ԥ��
			c10::IValue pred;
			try
			{
				pred = model({ map });
			}
			catch (const std::exception& e)
			{
				std::cerr << "[DeepEE] Error predicting: " << e.what() << std::endl;
				return NAN;
			}
			return pred.toTensor().reshape({ 1 }).item().toFloat();
		}

		c10::DeviceType get_device()
		{
			return device;
		}
	};
}