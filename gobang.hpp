#pragma once
/*****************************************************************************
* gobang.hpp
* 
* Gobang SDK
*****************************************************************************/

#include <iostream>
#include <array>
#include <vector>
#include <thread>

namespace gobang
{
	const short G_EMPTY = 0;
	const short G_BLACK = 1;
	const short G_WHITE = 2;

	//��ת��ɫ
	int Invert(int col)
	{
		if (col == G_WHITE)return G_BLACK;
		else if (col == G_BLACK)return G_WHITE;
		return col;
	}

	//����
	class Board
	{
	private:
		int find_fbd(int x, int y)
		{
			int res = 0, b = G_BLACK;
			//����
			res += find_at("-ooo-", { x,y }, b);
			//���� + ����
			res += find_at("oooo-", { x,y }, b) - find_at("-oooo-", { x,y }, b);
			//Ƕ��
			res += find_at("ooo-o-", { x,y }, b) + find_at("-ooo-o", { x,y }, b) - find_at("-ooo-o-", { x,y }, b);
			res += find_at("oo-oo-", { x,y }, b) - find_at("-oo-oo-", { x,y }, b);
			return res;
		}

		static std::array<short, 15> null_array;

		//���ĳ���±��Ƿ��ڷ�Χ��
		bool safe(int x, int y)
		{
			return x >= 0 && x < 15 && y >= 0 && y < 15;
		}
	public:
		std::array<std::array<short, 15>, 15> map;

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
			bool match[4];
			for (int x = 0; x < 15; x++)
			{
				for (int y = 0; y < 15; y++)
				{
					for (bool& b : match)b = true;
					//�ڴ�λ�ò��ҽṹ
					for (int i = 0; i < str.size(); i++)
					{
						if (!safe(x + i, y)) match[0] = false;
						else match[0] &= map[x + i][y] == str[i];

						if (!safe(x, y + i)) match[1] = false;
						else match[1] &= map[x][y + i] == str[i];

						if (!safe(x + i, y + i)) match[2] = false;
						else match[2] &= map[x + i][y + i] == str[i];

						if (!safe(x + i, y - i)) match[3] = false;
						else match[3] &= map[x + i][y - i] == str[i];
					}
					for (bool b : match)if (b)res++;
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
		//�������е�ĳһλ��Ѱ��ĳһ���νṹ, ������Ŀ
		int find_at(const std::vector<int>& str, std::pair<int, int> where, bool recursion = false)
		{
			if (str.size() == 0)return 0;
			int x = where.first, y = where.second;
			//����
			int res = 0;
			bool match[4] = { true };
			//i ����ṹ����ʼλ��
			for (int i = 1 - str.size(); i <= 0; i++)
			{
				for (bool& b : match)b = true;
				//�ڴ�λ�ò��ҽṹ
				for (int j = 0; j < str.size(); j++)
				{
					if (!safe(x + i + j, y)) match[0] = false;
					else match[0] &= map[x + i + j][y] == str[j];

					if (!safe(x, y + i + j)) match[1] = false;
					else match[1] &= map[x][y + i + j] == str[j];

					if (!safe(x + i + j, y + i + j)) match[2] = false;
					else match[2] &= map[x + i + j][y + i + j] == str[j];

					if (!safe(x + i + j, y - i - j)) match[3] = false;
					else match[3] &= map[x + i + j][y - i - j] == str[j];
				}
				for (bool b : match)if (b)res++;
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
					res += find_at(trs, where, true);
				}
			}
			return res;
		}

		//��������Ѱ��ĳһ���νṹ, ������Ŀ
		//str: o��ʾ��ɫΪcol�����ӣ�x��ʾ��ɫ�����ӻ�߽磬-��ʾ��λ
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
		//�������е�ĳһλ��Ѱ��ĳһ���νṹ, ������Ŀ
		//str: o��ʾ��ɫΪcol�����ӣ�x��ʾ��ɫ�����ӻ�߽磬-��ʾ��λ
		int find_at(const std::string& str, std::pair<int, int> where, int col)
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
			return find_at(temp, where);
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
				int cnt0 = find_fbd(x, y);

				//����
				map[x][y] = col;

				//����
				if (find_at("oooooo", { x,y }, col) > 0)
				{
					//��ԭ
					map[x][y] = G_EMPTY;
					return false;
				}


				int cnt1 = find_fbd(x, y);

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

		std::array<short, 15>& operator[](unsigned i)
		{
			if (i < 15)return map[i];
			else return null_array;
		}

		friend std::ostream& operator<<(std::ostream& os, const Board bd)
		{
			for (int y = 0; y < 15; y++)
			{
				for (int x = 0; x < 15; x++)
				{
					auto piece = bd.map[x][y];
					if (piece == G_EMPTY)os << ". ";
					else if (piece == G_BLACK)os << "o ";
					else if (piece == G_WHITE)os << "x ";
				}
				os << std::endl;
			}
			return os;
		}
	};


	class Fish
	{
	public:
		virtual std::array<int, 2> play(Board bd, int self, bool forbidden = true) = 0;
	};
}