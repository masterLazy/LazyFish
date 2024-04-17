#include <Windows.h>
#include <windowsx.h>
#include <atomic>

#include "../mLib/mGraphics.h"

#include "gobang.hpp"
#include "basic_fish.hpp"

using namespace mlib;
using namespace gobang;
using namespace std;

//五子棋
Board board;
BasicFish bf;
int current_player = G_BLACK;

//下棋AI的线程
#define SG_NULL	0
#define SG_PLAY	1
#define SG_END	-1
atomic<int> signal = SG_NULL;
array<int, 2> predict = { -1,-1 };
void th2func()
{
	while (signal != SG_END)
	{
		if (signal == SG_PLAY)
		{
			predict = bf.play(board, current_player, true);
			//signal = SG_NULL;
			board[predict[0]][predict[1]] = current_player;
			current_player = Invert(current_player);
			if (board.is_end())
			{
				board.clear();
			}
		}
	}
}

Graphics gfx;
int cxWindow = 600, cyWindow = 700;

void draw_board()
{
	//明确绘图范围
	float gap = cxWindow / 16.0;
	Rect rect = { gap,cyWindow - 15 * gap,cxWindow - gap,cyWindow - gap };
	//绘制通知
	wstring notice;
	if (current_player == G_BLACK)notice = L"Blank's turn.";
	else notice = L"White's turn.";
	if (signal == SG_PLAY)notice += L" Predicting. . .";
	gfx.draw_text(cxWindow / 2.0, rect.top / 2.0, notice, gfx.brush(10, 10, 10), FontAlign::middle);
	//绘制格子
	for (int i = 0; i < 15; i++)
	{
		gfx.draw_line(rect.left + gap * i, rect.top, rect.left + gap * i, rect.bottom, gfx.brush(10, 10, 10));
		gfx.draw_line(rect.left, rect.top + gap * i, rect.right, rect.top + gap * i, gfx.brush(10, 10, 10));
	}
	//绘制棋子
	for (int x = 0; x < 15; x++)
	{
		for (int y = 0; y < 15; y++)
		{
			if (board[x][y] != G_EMPTY)
			{
				gfx.draw_circle(rect.left + gap * x, rect.top + gap * y, gap * 0.4, gfx.brush(0, 0, 0));
				gfx.fill_circle(rect.left + gap * x, rect.top + gap * y, gap * 0.4,
					board[x][y] == G_BLACK ? gfx.brush(50, 50, 50) : gfx.brush(250, 250, 250));
			}
			/*else if (!board.is_able(x, y, G_BLACK))
			{
				gfx.draw_line(rect.left + gap * x - gap * 0.4, rect.top + gap * y - gap * 0.4,
					rect.left + gap * x + gap * 0.4, rect.top + gap * y + gap * 0.4, gfx.brush(250, 100, 100), 4);
				gfx.draw_line(rect.left + gap * x + gap * 0.4, rect.top + gap * y - gap * 0.4,
					rect.left + gap * x - gap * 0.4, rect.top + gap * y + gap * 0.4, gfx.brush(250, 100, 200), 4);
			}*/
			else if (x == predict[0] && y == predict[1])
			{
				gfx.fill_rectangle(rect.left + gap * x - gap * 0.2, rect.top + gap * y - gap * 0.2,
					rect.left + gap * x + gap * 0.2, rect.top + gap * y + gap * 0.2, gfx.brush(100, 100, 240));
			}
			/*if (abs(gfx.m_x - rect.left - gap * x) + abs(gfx.m_y - rect.top - gap * y) < gap / 2
				&& board.is_able(x, y, current_player))
			{
				gfx.draw_line(rect.left + gap * x - gap * 0.4, rect.top + gap * y - gap * 0.4,
					rect.left + gap * x + gap * 0.4, rect.top + gap * y + gap * 0.4, gfx.brush(100, 100, 200), 4);
				gfx.draw_line(rect.left + gap * x + gap * 0.4, rect.top + gap * y - gap * 0.4,
					rect.left + gap * x - gap * 0.4, rect.top + gap * y + gap * 0.4, gfx.brush(100, 100, 200), 4);
			}*/
		}
	}
}


LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow)
{
	//m_hInstance = hInstance;//保存一下实例句柄

	//第二线程
	thread th2(th2func);
	th2.detach();

	wchar_t szAppName[] = L"五子棋";
	HWND hWnd;
	MSG msg;
	WNDCLASS wndclass;

	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(hInstance, IDI_WINLOGO);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);//有控件时，要改成这样，更美观
	wndclass.lpszMenuName = NULL;
	wndclass.lpszClassName = szAppName;

	if (!RegisterClass(&wndclass))
	{
		MessageBox(NULL, L"窗口注册失败！", szAppName, MB_ICONERROR);
		return 0;
	}

	hWnd = CreateWindow(szAppName,
		szAppName,//窗口的标题
		WS_CAPTION | WS_SYSMENU,
		CW_USEDEFAULT,//窗口的x坐标
		CW_USEDEFAULT,//窗口的y坐标
		cxWindow,  //窗口的宽
		cyWindow, //窗口的高
		NULL,
		NULL,
		hInstance,
		NULL);

	ShowWindow(hWnd, iCmdShow);
	UpdateWindow(hWnd);

	while (true)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				break;
			}
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	return msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	//自动操作
	if (gfx.proc(hWnd, message, wParam, lParam))return 0;

	switch (message)
	{
	case WM_CREATE:
	{
		//修改窗口大小和位置
		RECT rect;
		GetClientRect(hWnd, (RECT*)&rect);
		MoveWindow(hWnd, 50, 50, cxWindow + (cxWindow - rect.right), cyWindow + (cyWindow - rect.bottom), false);
		return 0;
	}
	case WM_PAINT:
	{
		gfx.begin_draw();
		gfx.clear(RGB(240, 240, 240));
		draw_board();
		gfx.end_draw();
		return 0;
	}
	case WM_LBUTTONDOWN:
	{
		//明确范围
		float gap = cxWindow / 16.0;
		Rect rect = { gap,cyWindow - 15 * gap,cxWindow - gap,cyWindow - gap };
		//落棋
		for (int x = 0; x < 15; x++)
		{
			for (int y = 0; y < 15; y++)
			{
				if (abs(gfx.m_x - rect.left - gap * x) + abs(gfx.m_y - rect.top - gap * y) < gap / 2)
				{
					board[x][y] = current_player;
					current_player = Invert(current_player);
					signal = SG_PLAY;
				}
			}
		}
		return 0;
	}
	case WM_DESTROY:
	{
		signal = SG_END;
		PostQuitMessage(0);
		return 0;
	}
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
}