#pragma once
/*****************************************************************************
* gobang.hpp
*
* NOTICE:
* Please define _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING at the begining
* of your source code.
*****************************************************************************/


#include <array>
#include <vector>
#include <thread>

//#define _SILENCE_AMP_DEPRECATION_WARNINGS
//#include <amp.h>

namespace gobang
{
	const int G_EMPTY = 0;
	const int G_BLACK = 1;
	const int G_WHITE = 2;

	//��ת��ɫ
	int Invert(int p)
	{
		if (p == G_WHITE)return G_BLACK;
		else if (p == G_BLACK)return G_WHITE;
		return p;
	}

	//����
	class Board
	{
	private:
		int find_fbd()
		{
			int res = 0, b = G_BLACK;
			//����
			res += find("-ooo-", b);
			//���� + ����
			res += find("oooo-", b) - find("-oooo-", b);
			//Ƕ��
			res += find("ooo-o-", b) + find("-ooo-o", b) - find("-ooo-o-", b);
			res += find("oo-oo-", b) - find("-oo-oo-", b);
			return res;
		}

		std::array<int, 15> null_array;
	public:
		std::array<std::array<int, 15>, 15> map;

		Board()
		{
			clear();
		}

		bool empty()
		{
			bool res = true;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					res &= map[x][y] == G_EMPTY;
				}
			}
			return res;
		}
		void clear()
		{
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					map[x][y] = G_EMPTY;
				}
			}
		}

		//��������Ѱ��ĳһ���νṹ, ������Ŀ
		int find(const std::vector<int>& str, bool recursion = false)
		{
			if (str.size() == 0)return 0;
			//����
			int res = 0;
			bool ok_0, ok_1, ok_2, ok_3;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					ok_0 = ok_1 = ok_2 = ok_3 = true;
					//�ڴ�λ�ò��ҽṹ
					for (int i = 0; i < str.size(); i++)
					{
						if (x + i >= 15) ok_0 = false;
						else ok_0 &= map[x + i][y] == str[i];

						if (y + i >= 15) ok_1 = false;
						else ok_1 &= map[x][y + i] == str[i];

						if (x + i >= 15 || y + i >= 15) ok_2 = false;
						else ok_2 &= map[x + i][y + i] == str[i];

						if (x + i >= 15 || y - i < 0) ok_3 = false;
						else ok_3 &= map[x + i][y - i] == str[i];
					}
					if (ok_0)res++;
					if (ok_1)res++;
					if (ok_2)res++;
					if (ok_3)res++;
				}
			}
			//������Գƣ���Ҫ������һ��
			if (!recursion)
			{
				bool symmetry = true;//�Ƿ�Գ�
				for (int i = 0; i < str.size() / 2; i++)
				{
					symmetry &= str[i] == str[str.size() - i - 1];
				}
				if (!symmetry)
				{
					std::vector<int> trs(str.size());
					for (int i = 0; i < str.size(); i++)
					{
						trs[i] = str[str.size() - i - 1];
					}
					res += find(trs, true);
				}
			}
			return res;
		}
#if def AMP_H
		int find_(std::vector<int> str, bool recursion = false)
		{
			using namespace Concurrency;
			if (str.size() == 0)return 0;
			//׼��
			//std::vector<int> res(15 * 15);
			int res[15 * 15] = { 0 };
			//չ��map
			std::vector<int> map_;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)map_.push_back(map[x][y]);
			}
			array_view<int, 2> res_gpu(15, 15, res);
			array_view<int, 2> map_gpu(15, 15, map_);
			array_view<int, 1> str_gpu(str.size(), str);
			res_gpu.discard_data();
			//GPU��������
			/*parallel_for_each(
				res_gpu.extent,
				[=](index<2> idx) restrict(amp)
				{
					bool ok_0, ok_1, ok_2, ok_3;
					ok_0 = ok_1 = ok_2 = ok_3 = true;
					int x = idx[0], y = idx[1];
					//�ڴ�λ�ò��ҽṹ
					for (int i = 0; i < str_gpu.extent[0]; i++)
					{
						if (x + i >= 15) ok_0 = false;
						else ok_0 &= map_gpu[x + i][y] == str_gpu[i];

						if (y + i >= 15) ok_1 = false;
						else ok_1 &= map_gpu[x][y + i] == str_gpu[i];

						if (x + i >= 15 || y + i >= 15) ok_2 = false;
						else ok_2 &= map_gpu[x + i][y + i] == str_gpu[i];

						if (x + i >= 15 || y - i < 0) ok_3 = false;
						else ok_3 &= map_gpu[x + i][y - i] == str_gpu[i];
					}
					if (ok_0)res_gpu[idx]++;
					if (ok_1)res_gpu[idx]++;
					if (ok_2)res_gpu[idx]++;
					if (ok_3)res_gpu[idx]++;
				}
			);*/
			res_gpu.synchronize();
			int res_ = 0;
			for (int i = 0; i < 15 * 15; i++)
			{
				res_ += res[i];
			}
			//������Գƣ���Ҫ������һ��
			if (!recursion)
			{
				bool symmetry = true;//�Ƿ�Գ�
				for (int i = 0; i < str.size() / 2; i++)
				{
					symmetry &= str[i] == str[str.size() - i - 1];
				}
				if (!symmetry)
				{
					std::vector<int> trs(str.size());
					for (int i = 0; i < str.size(); i++)
					{
						trs[i] = str[str.size() - i - 1];
					}
					res_ += find(trs, true);
				}
			}
			return res_;
		}
#endif
		//��������Ѱ��ĳһ���νṹ, ������Ŀ
		//str: o��ʾ��ɫΪp�����ӣ�x��ʾ��ɫ�����ӻ�߽磬-��ʾ��λ
		int find(const std::string& str, int col)
		{
			std::vector<int> temp;
			for (char c : str)
			{
				switch (c)
				{
				case 'o':
				case 'O':
					temp.push_back(col);
					break;

				case 'x':
				case 'X':
					temp.push_back(Invert(col));
					break;

				case '-':
					temp.push_back(G_EMPTY);
					break;
				}
			}
			if (temp.size() > 15)temp.resize(15);
			return find(temp);
		}

		//��ȡ��ʤ��(��û�У�����0)
		int get_winner()
		{
			if (find("ooooo", G_WHITE) > 0)return G_WHITE;
			if (find("ooooo", G_BLACK) > 0)return G_BLACK;
			return 0;
		}
		//�Ƿ�������
		bool is_full()
		{
			bool full = true;
			for (auto i : map)
			{
				for (auto j : i)
				{
					full &= j != G_EMPTY;
				}
			}
			return full;
		}
		//����Ƿ����(����������)
		bool is_end()
		{
			return get_winner() != G_EMPTY || is_full();
		}

		//�˴��Ƿ�������
		//col: �����ӵ���ɫ
		//fbd: �Ƿ����
		bool is_able(int x, int y, int col, bool fbd = true)
		{
			if (x < 0 || y < 0 || x >= 15 || y >= 15)return false;
			//�����������
			if (map[x][y] != G_EMPTY)return false;

			//����
			if (col == G_BLACK && fbd == true)
			{
				int cnt0 = find_fbd();

				//����
				map[x][y] = col;

				//����
				if (find("oooooo", col) > 0)
				{
					//��ԭ
					map[x][y] = G_EMPTY;
					return false;
				}


				int cnt1 = find_fbd();

				if (cnt1 - cnt0 > 1)
				{
					map[x][y] = G_EMPTY;
					return false;
				}

				//��ԭ
				map[x][y] = G_EMPTY;
			}

			return true;
		}

		std::array<int, 15>& operator[](unsigned i)
		{
			if (i < 15)return map[i];
			else return null_array;
		}
	};


	class Fish
	{
	public:
		virtual std::array<int, 2> play(Board bd, int self, bool forbidden = true) = 0;
	};
}