#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include "../mLib/mFunction.h"

#include "gobang.hpp"
#include "basic_fish.hpp"

using namespace gobang;
using namespace std;

//五子棋
Board board;
BasicFish bf;

atomic<int> total = 0;

//要制造的数据集的大小
void make(int len, int number)
{
	ofstream xout(("x_" + to_string(number)).c_str());
	ofstream yout(("y_" + to_string(number)).c_str());
	array<int, 2> pred;
	int cnt = 0;//已保存的游戏局数
	int t = 0;//当前局步数
	//缓存
	string x_temp[2], y_temp[2];
	//迭代棋局
	int current_player = G_BLACK;
	while (cnt < len)
	{
		//写入x
		for (int x = 0; x < 15; x++)
		{
			for (int y = 0; y < 15; y++)
			{
				if (board[x][y] == current_player) x_temp[current_player - 1] += "0.5";
				else if (board[x][y] == G_EMPTY) x_temp[current_player - 1] += "0.0";
				else x_temp[current_player - 1] += "1.0";
				if (x * y < 14 * 14)
				{
					x_temp[current_player - 1] += ",";
				}
			}
		}
		x_temp[current_player - 1] += "\n";
		//下棋
		if (board.empty())
		{
			pred = { rand() % 13 + 1,rand() % 13 + 1 };
		}
		else
		{
			pred = bf.play(board, current_player);
		}
		//写入y
		for (int x = 0; x < 15; x++)
		{
			for (int y = 0; y < 15; y++)
			{
				if (pred[0] == x && pred[1] == y) y_temp[current_player - 1] += "1";
				else y_temp[current_player - 1] += "0";
				if (x * y < 14 * 14)
				{
					y_temp[current_player - 1] += ",";
				}
			}
		}
		y_temp[current_player - 1] += "\n";
		//着子，切换当前玩家
		board[pred[0]][pred[1]] = current_player;
		current_player = Invert(current_player);
		//看看是不是结束了
		if (board.is_end())
		{
			int winner = board.get_winner();
			if (winner != 0)
			{
				xout << x_temp[winner - 1];
				yout << y_temp[winner - 1];
			}

			x_temp[0].clear();
			x_temp[1].clear();
			y_temp[0].clear();
			y_temp[1].clear();

			board.clear();
			current_player = G_BLACK;
			cnt++;
			total++;
		}
	}
}

int each_thread_goal = 2000;

int main()
{
	ios::sync_with_stdio(false);
	srand(time(NULL));

	vector<thread> threads;

	for (int i = 0; i < 10; i++)
	{
		threads.push_back(thread(make, each_thread_goal, i));
		threads.back().detach();
	}

	clock_t timer = clock();
	while (total < 10 * each_thread_goal)
	{
		cout << "\r";
		mlib::printPB(total / (10.0 * each_thread_goal), mlib::PBStyle::block);
		cout << " " << total << "/" << 10 * each_thread_goal
			<< ", avg " << float(clock() - timer) / total / 1000 << " s/game"
			<< ", ert " << float(clock() - timer) / total / 1000 * (10 * each_thread_goal - total) / 60 << " min";

		Sleep(100);
	}

	return 0;
}