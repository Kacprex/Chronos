#pragma once
#include "board.h"
#include <vector>

namespace chronos {

// Internal pseudo move generators (implemented in movegen.cpp)
void gen_pseudo(const Board& b, std::vector<Move>& out, bool captures_only);

} // namespace chronos
