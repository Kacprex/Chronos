#pragma once
#include "types.h"
#include <array>

namespace chronos {

struct Zobrist {
    std::array<std::array<U64, 64>, 13> piece_sq{}; // [piece][sq], piece index uses Piece enum (0..12)
    std::array<U64, 16> castling{};                 // castling rights 0..15
    std::array<U64, 9> ep_file{};                   // ep file 0..7 or 8=none
    U64 side_to_move{};
};

const Zobrist& zobrist();

} // namespace chronos
