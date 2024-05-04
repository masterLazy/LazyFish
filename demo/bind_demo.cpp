//务必把这几个 #include 放在开头
#include "gobang.hpp"
#include "basic_fish.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#pragma comment(lib,"python3.lib")
#pragma comment(lib,"python310.lib")

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(gobang, m)
{
	m.doc() = "Gobang API";
	m.attr("EMPTY") = py::cast(gobang::G_EMPTY);
	m.attr("BLACK") = py::cast(gobang::G_BLACK);
	m.attr("WHITE") = py::cast(gobang::G_WHITE);
	m.def("invert", &gobang::Invert, "col"_a, "Invert the color");
	py::class_<gobang::Board>(m, "Board")
		.def(py::init())
		.def("empty", &gobang::Board::empty, "If the board is empty")
		.def("clear", &gobang::Board::clear, "Clear the board")
		.def("find", py::overload_cast<const std::vector<int>&, bool>(&gobang::Board::find),
			"str"_a, "recursion"_a = false, "Find structure")
		.def("find", py::overload_cast<const std::string&, int>(&gobang::Board::find),
			"str"_a, "col"_a, "Find structure\n\"o\"=\"self, \"x\"=enemy, \"-\"=empty")
		.def("get_winner", &gobang::Board::get_winner)
		.def("is_full", &gobang::Board::is_full)
		.def("is_end", &gobang::Board::is_end)
		.def("is_able", &gobang::Board::is_able,
			"x"_a, "y"_a, "col"_a, "fbd"_a = true)
		//为 Python 定制的函数
		.def("__getitem__", [](gobang::Board& bd, std::pair<int, int> idx)
			{
				return bd[idx.first][idx.second];
			})
		.def("__setitem__", [](gobang::Board& bd, std::pair<int, int> idx, int value)
			{
				bd[idx.first][idx.second] = value;
			})
		.def("as_numpy", [](gobang::Board& bd)
			{
				py::array_t<int> arr({ 15,15 }, { sizeof(int), 15 * sizeof(int) });
				auto buf = arr.request();
				auto ptr = (int*)buf.ptr;
				for (int x = 0; x < 15; x++)
				{
					for (int y = 0; y < 15; y++)
					{
						auto element = ptr + x * 15 + y;
						*element = bd[x][y];
					}
				}
				return arr;
			});
}