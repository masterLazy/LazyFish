#pragma once
/*****************************************************************************
* basic_fish_2.hpp
*
* BasicFish 2.0
*****************************************************************************/

#include "gobang.hpp"
#include <random>

#ifndef max
inline float max(float a, float b)
{
	return a > b ? a : b;
}
#endif

namespace gobang
{
	class BasicFish :Fish
	{
	private:
		int try_find(Board bd, int x, int y, int self, std::string str)
		{
			int h0 = bd[x][y];
			int cnt0 = bd.find_at(str, { x,y }, self);
			bd[x][y] = self;
			int cnt1 = bd.find_at(str, { x,y }, self);
			bd[x][y] = h0;
			return cnt1 - cnt0;
		}
		//���ĳ���±��Ƿ��ڷ�Χ��
		bool safe(int x, int y)
		{
			return x >= 0 && x < 15 && y >= 0 && y < 15;
		}
	public:
		//[0,1]��Ԥ�����
		std::array<std::array<float, 15>, 15> predict(Board bd, int self, float noise = 0)
		{
			//���ڼ�¼��ÿ��������ĺ��ʶ�
			std::array<std::array<float, 15>, 15> m;
			//��ʼ��
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					m[x][y] = 0;
				}
			}
			int enemy = Invert(self);
			int inf = 1000;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd[x][y] == G_EMPTY)
					{
						//ʤ

						/*Att*/if (try_find(bd, x, y, self, "ooooo") > 0)
						{
							m[x][y] = inf;
							x = y = inf;
							continue;
						}
						/*Dfs*/if (try_find(bd, x, y, enemy, "ooooo") > 0)
						{
							m[x][y] = inf - 1;
							continue;
						}

						//��ʤ

						/*Att*/if (try_find(bd, x, y, self, "-ooo-") +
							try_find(bd, x, y, self, "-oo-o-") +
							try_find(bd, x, y, self, "-oooo") > 2
							||
							try_find(bd, x, y, self, "-oooo-") > 0)
						{
							m[x][y] = inf - 2;
							continue;
						}
						/*Dfs*/if (try_find(bd, x, y, enemy, "-ooo-") +
							try_find(bd, x, y, enemy, "-oo-o-") +
							try_find(bd, x, y, enemy, "-oooo") > 2
							||
							try_find(bd, x, y, enemy, "-oooo-") > 0)
						{
							m[x][y] = inf - 3;
							continue;
						}

						//��ǿ���ŷ�

						/*Att*/if (try_find(bd, x, y, self, "-ooo-") +
							try_find(bd, x, y, self, "-oo-o-") > 0)
						{
							m[x][y] += 8;
						}
						/*Dfs*/if (try_find(bd, x, y, enemy, "-ooo-") > 0)
						{
							m[x][y] += 4;
						}

						/*Att*/if (
							try_find(bd, x, y, self, "-oo--") +
							try_find(bd, x, y, self, "-o-o-") > 0)
						{
							m[x][y] += 2;
						}
						/*Dfs*/if (try_find(bd, x, y, enemy, "-oo--") > 0)
						{
							m[x][y] += 1;
						}
					}
				}
			}

			//�������
			if (noise != 0)
			{
				std::random_device rd;
				std::default_random_engine generator(rd());
				std::normal_distribution<float> distribution(0.0, noise);
				for (int x = 0; x < 15; x++)
				{
					for (int y = 0; y < 15; y++)
					{
						m[x][y] += round(distribution(generator));
					}
				}
			}

			return m;
		}
		//��ȡ����λ��
		std::array<int, 2> play(Board bd, int self, float noise, bool fbd = true)
		{
			auto pred = predict(bd, self, noise);

			//������ʵĵط�����
			int _x = -1, _y = -1;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd.is_able(x, y, self, fbd) &&
						(_x == -1 || pred[x][y] > pred[_x][_y]))
					{
						_x = x;
						_y = y;
					}
				}
			}

			return { _x,_y };
		}
		std::array<int, 2> play(Board bd, int self, bool fbd = true)
		{
			return play(bd, self, 0.15, fbd);
		}
	};
}