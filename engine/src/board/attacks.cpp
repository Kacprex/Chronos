#include "chronos/board/attacks.h"
#include "chronos/util/bitops.h"
#include <array>

namespace chronos {

static std::array<U64, 64> KN{};
static std::array<U64, 64> KG{};
static std::array<std::array<U64, 64>, 2> PA{};

static bool on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

void init_attacks() {
    for (int sq = 0; sq < 64; ++sq) {
        int f = file_of(sq), r = rank_of(sq);
        U64 k = 0, n = 0;

        // King
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) continue;
                int nf = f + df, nr = r + dr;
                if (on_board(nf, nr)) k |= bit(make_square(nf, nr));
            }

        // Knight
        const int d[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
        for (auto& dv : d) {
            int nf = f + dv[0], nr = r + dv[1];
            if (on_board(nf, nr)) n |= bit(make_square(nf, nr));
        }

        KN[sq] = n;
        KG[sq] = k;

        // Pawns: attacks *from* sq
        U64 w = 0, b = 0;
        if (on_board(f - 1, r + 1)) w |= bit(make_square(f - 1, r + 1));
        if (on_board(f + 1, r + 1)) w |= bit(make_square(f + 1, r + 1));
        if (on_board(f - 1, r - 1)) b |= bit(make_square(f - 1, r - 1));
        if (on_board(f + 1, r - 1)) b |= bit(make_square(f + 1, r - 1));
        PA[WHITE][sq] = w;
        PA[BLACK][sq] = b;
    }
}

U64 knight_attacks(Square sq) { return KN[sq]; }
U64 king_attacks(Square sq) { return KG[sq]; }
U64 pawn_attacks(Color c, Square sq) { return PA[c][sq]; }

static U64 ray(Square sq, int df, int dr, U64 occ) {
    int f = file_of(sq), r = rank_of(sq);
    U64 a = 0;
    while (true) {
        f += df; r += dr;
        if (!on_board(f, r)) break;
        Square ns = make_square(f, r);
        a |= bit(ns);
        if (occ & bit(ns)) break;
    }
    return a;
}

U64 bishop_attacks(Square sq, U64 occ) {
    return ray(sq, 1, 1, occ) | ray(sq, -1, 1, occ) | ray(sq, 1, -1, occ) | ray(sq, -1, -1, occ);
}
U64 rook_attacks(Square sq, U64 occ) {
    return ray(sq, 1, 0, occ) | ray(sq, -1, 0, occ) | ray(sq, 0, 1, occ) | ray(sq, 0, -1, occ);
}
U64 queen_attacks(Square sq, U64 occ) { return bishop_attacks(sq, occ) | rook_attacks(sq, occ); }

} // namespace chronos
