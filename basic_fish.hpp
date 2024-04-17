#pragma once
/*****************************************************************************
* basic_fish.hpp
*****************************************************************************/

#include "gobang.hpp"

#ifndef max
template<typename T>
inline T max(T a, T b)
{
	return a > b ? a : b;
}
#endif

namespace gobang
{
	class BasicFish :Fish
	{
	private:
		int try_find(Board bd, int x, int y, int col, std::string str)
		{
			int h0 = bd[x][y];
			int cnt0 = bd.find(str, col);
			bd[x][y] = col;
			int cnt1 = bd.find(str, col);
			bd[x][y] = h0;
			return cnt1 - cnt0;
		}
		//检测某个下标是否在范围内
		bool safe(int x, int y)
		{
			return x >= 0 && x < 15 && y >= 0 && y < 15;
		}
	public:
		std::array<int, 2> play(Board bd, int self, bool fbd = true)
		{
			//用于记录在每个点下棋的合适度
			int m[15][15];
			//初始化
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					m[x][y] = 0;
				}
			}
			int enemy = Invert(self);
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					//攻击
					if (bd.is_able(x, y, self, fbd))
					{
						//我方在这里下子后连五: 8
						if (try_find(bd, x, y, self, "ooooo") > 0)
						{
							m[x][y] = max(m[x][y], 8);
						}

						//我方在这里下子后连四: 6
						if (try_find(bd, x, y, self, "oooo") > 0)
						{
							m[x][y] = max(m[x][y], 6);
						}

						//我方在这里下子后连三: 4
						//我方在这里下子后连多三: 6
						int cnt = try_find(bd, x, y, self, "ooo");
						if (cnt > 1)
						{
							m[x][y] = max(m[x][y], 6);
						}
						else if (cnt == 1)
						{
							m[x][y] = max(m[x][y], 4);
						}

						//我方在这里下子后连二: 2
						if (try_find(bd, x, y, self, "oo") > 1)
						{
							m[x][y] = max(m[x][y], 2);
						}
					}
					//防守
					if (bd[x][y] == enemy)
					{
						//活一(可变为二): 1
						if (safe(x - 1, y - 1) && bd[x - 1][y - 1] == 0 && m[x - 1][y - 1] < 1)
						{
							m[x - 1][y - 1] = 1;
						}
						if (safe(x, y - 1) && bd[x][y - 1] == 0 && m[x][y - 1] < 1)
						{
							m[x][y - 1] = 1;
						}
						if (safe(x + 1, y - 1) && bd[x + 1][y - 1] == 0 && m[x + 1][y - 1] < 1)
						{
							m[x + 1][y - 1] = 1;
						}
						if (safe(x + 1, y) && bd[x + 1][y] == 0 && m[x + 1][y] < 1)
						{
							m[x + 1][y] = 1;
						}
						if (safe(x + 1, y + 1) && bd[x + 1][y + 1] == 0 && m[x + 1][y + 1] < 1)
						{
							m[x + 1][y + 1] = 1;
						}
						if (safe(x, y + 1) && bd[x][y + 1] == 0 && m[x][y + 1] < 1)
						{
							m[x][y + 1] = 1;
						}
						if (safe(x - 1, y + 1) && bd[x - 1][y + 1] == 0 && m[x - 1][y + 1] < 1)
						{
							m[x - 1][y + 1] = 1;
						}
						if (safe(x - 1, y) && bd[x - 1][y] == 0 && m[x - 1][y] < 1)
						{
							m[x - 1][y] = 1;
						}

						//活二(可变为三): 3
						//活二(可变为多三): 5
						//"一"
						if (safe(x - 1, y) && bd[x - 1][y] == 0 &&
							bd[x][y] == enemy &&
							safe(x + 1, y) && bd[x + 1][y] == enemy &&
							safe(x + 2, y) && bd[x + 2][y] == 0)
						{
							if (safe(x - 2, y) && bd[x - 2][y] == 0)
							{
								if (m[x - 1][y] < 3)m[x - 1][y] = 3;
								else if (m[x][y - 1] == 3)m[x - 1][y] = 5;
							}
							if (safe(x + 3, y) && bd[x + 3][y] == 0 && m[x + 2][y] < 3)
							{
								if (m[x + 2][y] < 3)m[x + 2][y] = 3;
								else if (m[x][y + 2] == 3)m[x + 2][y] = 5;
							}
						}
						//"|"
						if (safe(x, y - 1) && bd[x][y - 1] == 0 &&
							bd[x][y] == enemy &&
							safe(x, y + 1) && bd[x][y + 1] == enemy &&
							safe(x, y + 2) && bd[x][y + 2] == 0)
						{
							if (safe(x, y - 2) && bd[x][y - 2] == 0)
							{
								if (m[x][y - 1] < 3)m[x][y - 1] = 3;
								else if (m[x][y - 1] == 3)m[x][y - 1] = 5;
							}
							if (safe(x, y + 3) && bd[x][y + 3] == 0)
							{
								if (m[x][y + 2] < 3)m[x][y + 2] = 3;
								else if (m[x][y + 2] == 3)m[x][y + 2] = 5;
							}
						}
						//"/"
						if (safe(x - 1, y + 1) && bd[x - 1][y + 1] == 0 &&
							bd[x][y] == enemy &&
							safe(x + 1, y - 1) && bd[x + 1][y - 1] == enemy &&
							safe(x + 2, y - 2) && bd[x + 2][y - 2] == 0)
						{
							if (safe(x - 2, y + 2) && bd[x - 2][y + 2] == 0 && m[x - 1][y + 1] < 3)
							{
								if (m[x - 1][y + 1] < 3)m[x - 1][y + 1] = 3;
								else if (m[x - 1][y + 1] == 3)m[x - 1][y + 1] = 5;
							}
							if (safe(x + 3, y - 3) && bd[x + 3][y - 3] == 0 && m[x + 2][y - 2] < 3)
							{
								if (m[x + 2][y - 2] < 3)m[x + 2][y - 2] = 3;
								else if (m[x + 2][y - 2] == 3)m[x + 2][y - 2] = 5;
							}
						}
						//"\"
						if (safe(x - 1, y - 1) && bd[x - 1][y - 1] == 0 &&
							bd[x][y] == enemy &&
							safe(x + 1, y + 1) && bd[x + 1][y + 1] == enemy &&
							safe(x + 2, y + 2) && bd[x + 2][y + 2] == 0)
						{
							if (safe(x - 2, y - 2) && bd[x - 2][y + 2] == 0 && m[x - 1][y - 1] < 3)
							{
								if (m[x - 1][y - 1] < 3)m[x - 1][y - 1] = 3;
								else if (m[x - 1][y - 1] == 3)m[x - 1][y - 1] = 5;
							}
							if (safe(x + 3, y + 3) && bd[x + 3][y - 3] == 0 && m[x + 2][y + 2] < 3)
							{
								if (m[x + 2][y + 2] < 3)m[x + 2][y + 2] = 3;
								else if (m[x + 2][y + 2] == 3)m[x + 2][y + 2] = 5;
							}
						}
						//活三(可变为四): 5
						//活三(可变为多四): 6
						//"一"
						if (safe(x - 2, y) && bd[x - 2][y] == 0 &&
							safe(x - 1, y) && bd[x - 1][y] == enemy &&
							bd[x][y] == enemy &&
							safe(x + 1, y) && bd[x + 1][y] == enemy &&
							safe(x + 2, y) && bd[x + 2][y] == 0)
						{
							if (m[x - 2][y] < 5)m[x - 2][y] = 5;
							else if (m[x - 2][y] == 5)m[x - 2][y] = 6;
							if (m[x + 2][y] < 5)m[x + 2][y] = 5;
							else if (m[x + 2][y] == 5)m[x + 2][y] = 6;
						}
						//"|"
						if (safe(x, y - 2) && bd[x][y - 2] == 0 &&
							safe(x, y - 1) && bd[x][y - 1] == enemy &&
							bd[x][y] == enemy &&
							safe(x, y + 1) && bd[x][y + 1] == enemy &&
							safe(x, y + 2) && bd[x][y + 2] == 0)
						{
							if (m[x][y - 2] < 5)m[x][y - 2] = 5;
							else if (m[x][y - 2] == 5)m[x][y - 2] = 6;
							if (m[x][y + 2] < 5)m[x][y + 2] = 5;
							else if (m[x][y + 2] == 5)m[x][y + 2] = 6;
						}
						//"/"
						if (safe(x - 2, y + 2) && bd[x - 2][y + 2] == 0 &&
							safe(x - 1, y + 1) && bd[x - 1][y + 1] == enemy &&
							bd[x][y] == enemy &&
							safe(x + 1, y - 1) && bd[x + 1][y - 1] == enemy &&
							safe(x + 2, y - 2) && bd[x + 2][y - 2] == 0)
						{
							if (m[x - 2][y + 2] < 5)m[x - 2][y + 2] = 5;
							else if (m[x - 2][y + 2] == 5)m[x - 2][y + 2] = 6;
							if (m[x + 2][y - 2] < 5)m[x + 2][y - 2] = 5;
							else if (m[x + 2][y - 2] == 5)m[x + 2][y - 2] = 6;
						}
						//"\"
						if (safe(x - 2, y - 2) && bd[x - 2][y - 2] == 0 &&
							safe(x - 1, y - 1) && bd[x - 1][y - 1] == enemy &&
							bd[x][y] == enemy &&
							safe(x + 1, y + 1) && bd[x + 1][y + 1] == enemy &&
							safe(x + 2, y + 2) && bd[x + 2][y + 2] == 0)
						{
							if (m[x - 2][y - 2] < 5)m[x - 2][y - 2] = 5;
							else if (m[x - 2][y - 2] == 5)m[x - 2][y - 2] = 6;
							if (m[x + 2][y + 2] < 5)m[x + 2][y + 2] = 5;
							else if (m[x + 2][y + 2] == 5)m[x + 2][y + 2] = 6;
						}
					}
					if (bd[x][y] == 0)
					{
						//敌方在这里下棋后连五: 7
						bd[x][y] = enemy;
						if (bd.find("xxxxx", self) > 0 && m[x][y] < 7)m[x][y] = 7;
						bd[x][y] = 0;
						//敌方在这里下棋后形成"米"形中空型多活三: 5
						int count = 0;//记录活三的个数
						if (safe(x - 1, y) && bd[x - 1][y] == enemy &&
							safe(x + 1, y) && bd[x + 1][y] == enemy &&
							safe(x - 2, y) && bd[x - 2][y] == 0 &&
							safe(x + 2, y) && bd[x + 2][y] == 0)
						{
							count++;
						}
						if (safe(x, y - 1) && bd[x][y - 1] == enemy &&
							safe(x, y + 1) && bd[x][y + 1] == enemy &&
							safe(x, y - 2) && bd[x][y - 2] == 0 &&
							safe(x, y + 2) && bd[x][y + 2] == 0)
						{
							count++;
						}
						if (safe(x - 1, y - 1) && bd[x - 1][y - 1] == enemy &&
							safe(x + 1, y + 1) && bd[x + 1][y + 1] == enemy &&
							safe(x - 2, y - 2) && bd[x - 2][y - 2] == 0 &&
							safe(x + 2, y + 2) && bd[x + 2][y + 2] == 0)
						{
							count++;
						}
						if (safe(x - 1, y + 1) && bd[x - 1][y + 1] == enemy &&
							safe(x + 1, y - 1) && bd[x + 1][y - 1] == enemy &&
							safe(x - 2, y + 2) && bd[x - 2][y + 2] == 0 &&
							safe(x + 2, y - 2) && bd[x + 2][y - 2] == 0)
						{
							count++;
						}
						if (count >= 2)m[x][y] = 5;
					}
				}
			}

			//在最合适的地方下棋
			int _x = -1, _y = -1;
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					if (bd.is_able(x, y, self, fbd))
					{
						if (_x == -1 || m[x][y] > m[_x][_y])
						{
							_x = x;
							_y = y;
						}
						else if (m[x][y] == m[_x][_y] && rand() % 15 * 15 == 0)
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