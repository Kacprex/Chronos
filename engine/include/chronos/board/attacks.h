#pragma once
#include "../util/types.h"

namespace chronos {

void init_attacks();
U64 knight_attacks(Square sq);
U64 king_attacks(Square sq);
U64 pawn_attacks(Color c, Square sq);

// Sliding attacks (simple rays; MVP)
U64 bishop_attacks(Square sq, U64 occ);
U64 rook_attacks(Square sq, U64 occ);
U64 queen_attacks(Square sq, U64 occ);

} // namespace chronos
