#pragma once
#include "../util/types.h"
#include <string>

namespace chronos {

// Move encoding (32-bit):
// bits 0..5   from
// bits 6..11  to
// bits 12..15 promo (0 none, 1=N,2=B,3=R,4=Q)
// bits 16..23 flags
// Remaining unused.

enum MoveFlag : int {
    MF_NONE        = 0,
    MF_CAPTURE     = 1 << 0,
    MF_DOUBLE_PUSH = 1 << 1,
    MF_ENPASSANT   = 1 << 2,
    MF_CASTLE      = 1 << 3,
    MF_PROMOTION   = 1 << 4
};

using Move = std::uint32_t;

inline Move make_move(Square from, Square to, int promo, int flags) {
    return (Move(from) & 63)
         | ((Move(to) & 63) << 6)
         | ((Move(promo) & 15) << 12)
         | ((Move(flags) & 255) << 16);
}
inline Square move_from(Move m) { return Square(m & 63); }
inline Square move_to(Move m) { return Square((m >> 6) & 63); }
inline int move_promo(Move m) { return int((m >> 12) & 15); }
inline int move_flags(Move m) { return int((m >> 16) & 255); }

std::string move_to_uci(Move m);
Move uci_to_move(const std::string& uci, const class Board& b);

} // namespace chronos
