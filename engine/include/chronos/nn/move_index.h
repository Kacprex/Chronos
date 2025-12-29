#pragma once
#include "../board/board.h"

namespace chronos {

// AlphaZero-style action space: 8x8x73 = 4672
// For each from-square, 73 move-types:
// 0..55  : queen-like moves (8 dirs * 7 distances)
// 56..63 : knight moves (8)
// 64..72 : underpromotions (N,B,R) * (forward, capture-left, capture-right) = 9
static constexpr int MOVE_TYPES = 73;
static constexpr int MOVE_SPACE = 64 * MOVE_TYPES;

// Returns [0, MOVE_SPACE) or -1 if not representable.
int move_to_index(const Board& b, Move m);

} // namespace chronos
