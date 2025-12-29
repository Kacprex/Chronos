#include "chronos/nn/move_index.h"
#include "chronos/util/types.h"
#include <cmath>

namespace chronos {

static inline int file_of_sq(Square s) { return int(s) & 7; }
static inline int rank_of_sq(Square s) { return (int(s) >> 3) & 7; }

static constexpr int DIRS[8][2] = {
    { 0,  1}, // N
    { 0, -1}, // S
    { 1,  0}, // E
    {-1,  0}, // W
    { 1,  1}, // NE
    {-1,  1}, // NW
    { 1, -1}, // SE
    {-1, -1}, // SW
};

static constexpr int KN[8][2] = {
    { 1,  2}, { 2,  1}, { 2, -1}, { 1, -2},
    {-1, -2}, {-2, -1}, {-2,  1}, {-1,  2},
};

static int promo_group(int promo) {
    // promo: 1=N,2=B,3=R,4=Q (see move.h)
    if (promo == 1) return 0;
    if (promo == 2) return 1;
    if (promo == 3) return 2;
    return -1; // (Q or none)
}

int move_to_index(const Board& b, Move m) {
    Square from = move_from(m);
    Square to = move_to(m);
    int promo = move_promo(m);

    int ff = file_of_sq(from), fr = rank_of_sq(from);
    int tf = file_of_sq(to), tr = rank_of_sq(to);

    int df = tf - ff;
    int dr = tr - fr;

    int from_idx = int(from);
    if (from_idx < 0 || from_idx >= 64) return -1;

    // Underpromotion mapping (N,B,R only). Queen promotion is treated as a normal move type.
    Piece p = b.piece_on(from);
    bool is_pawn = (p == W_PAWN || p == B_PAWN);
    if (is_pawn) {
        int pg = promo_group(promo);
        if (pg >= 0) {
            Color stm = b.side_to_move();
            int fwd_dr = (stm == WHITE) ? 1 : -1;

            bool fwd = (df == 0 && dr == fwd_dr);
            bool capL = false;
            bool capR = false;
            if (stm == WHITE) {
                capL = (df == -1 && dr == 1);
                capR = (df ==  1 && dr == 1);
            } else {
                // black moves "down"; its left is +file
                capL = (df ==  1 && dr == -1);
                capR = (df == -1 && dr == -1);
            }

            int dir = fwd ? 0 : (capL ? 1 : (capR ? 2 : -1));
            if (dir < 0) return -1;

            int type = 64 + pg * 3 + dir; // 64..72
            return from_idx * MOVE_TYPES + type;
        }
    }

    // Knight moves
    for (int i = 0; i < 8; ++i) {
        if (df == KN[i][0] && dr == KN[i][1]) {
            int type = 56 + i; // 56..63
            return from_idx * MOVE_TYPES + type;
        }
    }

    // Queen-like moves (8 dirs * 7 distances)
    int adf = std::abs(df), adr = std::abs(dr);
    int dist = (adf > adr) ? adf : adr;
    if (dist < 1 || dist > 7) return -1;

    int sdf = (df == 0) ? 0 : (df / adf);
    int sdr = (dr == 0) ? 0 : (dr / adr);

    for (int d = 0; d < 8; ++d) {
        if (sdf == DIRS[d][0] && sdr == DIRS[d][1]) {
            int type = d * 7 + (dist - 1); // 0..55
            return from_idx * MOVE_TYPES + type;
        }
    }

    return -1;
}

} // namespace chronos
