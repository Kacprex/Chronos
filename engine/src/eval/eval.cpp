#include "chronos/eval/eval.h"
#include "chronos/util/bitops.h"
#include <array>

namespace chronos {

static const int PIECE_VALUE[PIECE_TYPE_NB] = {100, 320, 330, 500, 900, 0};

// Simple PST (middlegame-ish), symmetric
static const int PST[PIECE_TYPE_NB][64] = {
    // PAWN
    {
         0,  0,  0,  0,  0,  0,  0,  0,
        10, 10, 10, 10, 10, 10, 10, 10,
         6,  6,  7,  8,  8,  7,  6,  6,
         3,  3,  4,  6,  6,  4,  3,  3,
         1,  1,  2,  4,  4,  2,  1,  1,
         0,  0,  0,  2,  2,  0,  0,  0,
         0,  0,  0, -5, -5,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0
    },
    // KNIGHT
    {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    },
    // BISHOP
    {
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    },
    // ROOK
    {
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    },
    // QUEEN
    {
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    },
    // KING (middlegame)
    {
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    }
};

static int pst_for(Color c, PieceType pt, Square sq) {
    int s = (c == WHITE) ? sq : (sq ^ 56); // flip ranks for black
    return PST[pt][s];
}

EvalBreakdown evaluate_breakdown(const Board& b) {
    EvalBreakdown bd{};
    for (int c = 0; c < COLOR_NB; ++c) {
        Color col = Color(c);
        int sign = (col == WHITE) ? 1 : -1;

        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
            U64 bb = b.pieces_pt(col, PieceType(pt));
            while (bb) {
                Square sq = pop_lsb(bb);
                bd.material += sign * PIECE_VALUE[pt];
                bd.pst += sign * pst_for(col, PieceType(pt), sq);
            }
        }
    }

    // Perspective: return from side-to-move
    if (b.side_to_move() == BLACK) {
        bd.material = -bd.material;
        bd.pst = -bd.pst;
        bd.mobility = -bd.mobility;
        bd.king_safety = -bd.king_safety;
        bd.pawn_structure = -bd.pawn_structure;
        bd.space = -bd.space;
        bd.restriction = -bd.restriction;
    }
    return bd;
}

int evaluate(const Board& b) {
    return evaluate_breakdown(b).total();
}

} // namespace chronos
