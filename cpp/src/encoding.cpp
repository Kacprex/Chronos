#include "chronos/encoding.h"
#include <algorithm>

namespace chronos {

static int piece_plane(char p) {
    switch (p) {
        case 'P': return 0;
        case 'N': return 1;
        case 'B': return 2;
        case 'R': return 3;
        case 'Q': return 4;
        case 'K': return 5;
        case 'p': return 6;
        case 'n': return 7;
        case 'b': return 8;
        case 'r': return 9;
        case 'q': return 10;
        case 'k': return 11;
        default: return -1;
    }
}

std::array<float, INPUT_DIM> encode_planes(const Fen& f) {
    std::array<float, INPUT_DIM> x{};
    x.fill(0.0f);

    auto idx = [](int plane, int r, int c) {
        return plane * 64 + r * 8 + c;
    };

    // board planes: Fen::board uses ranks 8->1 in rows 0..7
    for (int r=0;r<8;r++) {
        for (int c=0;c<8;c++) {
            int pl = piece_plane(f.board[r][c]);
            if (pl >= 0) x[idx(pl, r, c)] = 1.0f;
        }
    }

    // side to move plane 12
    if (f.white_to_move) {
        for (int i=0;i<64;i++) x[12*64 + i] = 1.0f;
    }

    // castling planes
    if (f.castle_wk) for (int i=0;i<64;i++) x[13*64 + i] = 1.0f;
    if (f.castle_wq) for (int i=0;i<64;i++) x[14*64 + i] = 1.0f;
    if (f.castle_bk) for (int i=0;i<64;i++) x[15*64 + i] = 1.0f;
    if (f.castle_bq) for (int i=0;i<64;i++) x[16*64 + i] = 1.0f;

    // halfmove clock plane normalized
    float hm = std::min(std::max(f.halfmove_clock, 0), 100) / 100.0f;
    for (int i=0;i<64;i++) x[17*64 + i] = hm;

    return x;
}

} // namespace chronos
