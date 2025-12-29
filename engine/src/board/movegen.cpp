#include "chronos/board/movegen.h"
#include "chronos/board/attacks.h"
#include "chronos/util/bitops.h"

namespace chronos {

static bool is_own_piece(Piece p, Color stm) {
    return (stm == WHITE) ? is_white(p) : is_black(p);
}
static bool is_enemy_piece(Piece p, Color stm) {
    return (stm == WHITE) ? is_black(p) : is_white(p);
}

static void add_move(std::vector<Move>& out, Square from, Square to, int promo, int flags) {
    out.push_back(make_move(from, to, promo, flags));
}

static void gen_pawns(const Board& b, std::vector<Move>& out, bool captures_only) {
    Color stm = b.side_to_move();
    U64 occ = b.pieces_all();
    U64 ours = b.pieces(stm);
    U64 theirs = b.pieces(~stm);

    U64 pawns = b.pieces_pt(stm, PAWN);
    while (pawns) {
        Square from = pop_lsb(pawns);
        int f = file_of(from), r = rank_of(from);

        int dir = (stm == WHITE) ? 1 : -1;
        int start_rank = (stm == WHITE) ? 1 : 6;
        int promo_rank = (stm == WHITE) ? 6 : 1;
        int last_rank = (stm == WHITE) ? 7 : 0;

        // captures
        for (int df : {-1, 1}) {
            int nf = f + df;
            int nr = r + dir;
            if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
            Square to = make_square(nf, nr);
            bool is_ep = (to == b.ep_square());
            bool cap = (theirs & bit(to)) || is_ep;
            if (!cap) continue;

            int flags = MF_CAPTURE | (is_ep ? MF_ENPASSANT : MF_NONE);
            if (nr == last_rank) {
                // promotions
                add_move(out, from, to, 4, flags | MF_PROMOTION);
                add_move(out, from, to, 3, flags | MF_PROMOTION);
                add_move(out, from, to, 2, flags | MF_PROMOTION);
                add_move(out, from, to, 1, flags | MF_PROMOTION);
            } else {
                add_move(out, from, to, 0, flags);
            }
        }

        if (captures_only) continue;

        // single push
        int nr = r + dir;
        if (nr >= 0 && nr <= 7) {
            Square to = make_square(f, nr);
            if (!(occ & bit(to))) {
                if (r == promo_rank) {
                    add_move(out, from, to, 4, MF_PROMOTION);
                    add_move(out, from, to, 3, MF_PROMOTION);
                    add_move(out, from, to, 2, MF_PROMOTION);
                    add_move(out, from, to, 1, MF_PROMOTION);
                } else {
                    add_move(out, from, to, 0, MF_NONE);
                }

                // double push
                if (r == start_rank) {
                    Square to2 = make_square(f, r + 2*dir);
                    if (!(occ & bit(to2))) {
                        add_move(out, from, to2, 0, MF_DOUBLE_PUSH);
                    }
                }
            }
        }
    }
}

static void gen_knights(const Board& b, std::vector<Move>& out, bool captures_only) {
    Color stm = b.side_to_move();
    U64 ours = b.pieces(stm);
    U64 theirs = b.pieces(~stm);
    U64 kn = b.pieces_pt(stm, KNIGHT);
    while (kn) {
        Square from = pop_lsb(kn);
        U64 att = knight_attacks(from);
        U64 targets = captures_only ? (att & theirs) : (att & ~ours);
        U64 t = targets;
        while (t) {
            Square to = pop_lsb(t);
            int flags = (theirs & bit(to)) ? MF_CAPTURE : MF_NONE;
            add_move(out, from, to, 0, flags);
        }
    }
}

static void gen_kings(const Board& b, std::vector<Move>& out, bool captures_only) {
    Color stm = b.side_to_move();
    U64 ours = b.pieces(stm);
    U64 theirs = b.pieces(~stm);
    Square from = b.king_square(stm);
    U64 att = king_attacks(from);
    U64 targets = captures_only ? (att & theirs) : (att & ~ours);
    U64 t = targets;
    while (t) {
        Square to = pop_lsb(t);
        int flags = (theirs & bit(to)) ? MF_CAPTURE : MF_NONE;
        add_move(out, from, to, 0, flags);
    }
    if (captures_only) return;

    // Castling (pseudo; legality filtered by make())
    if (stm == WHITE) {
        if (b.castling_rights() & WK) {
            if (!(b.pieces_all() & (bit(make_square(5,0)) | bit(make_square(6,0))))
                && !b.is_square_attacked(make_square(4,0), BLACK)
                && !b.is_square_attacked(make_square(5,0), BLACK)
                && !b.is_square_attacked(make_square(6,0), BLACK)) {
                add_move(out, from, make_square(6,0), 0, MF_CASTLE);
            }
        }
        if (b.castling_rights() & WQ) {
            if (!(b.pieces_all() & (bit(make_square(1,0)) | bit(make_square(2,0)) | bit(make_square(3,0))))
                && !b.is_square_attacked(make_square(4,0), BLACK)
                && !b.is_square_attacked(make_square(3,0), BLACK)
                && !b.is_square_attacked(make_square(2,0), BLACK)) {
                add_move(out, from, make_square(2,0), 0, MF_CASTLE);
            }
        }
    } else {
        if (b.castling_rights() & BK) {
            if (!(b.pieces_all() & (bit(make_square(5,7)) | bit(make_square(6,7))))
                && !b.is_square_attacked(make_square(4,7), WHITE)
                && !b.is_square_attacked(make_square(5,7), WHITE)
                && !b.is_square_attacked(make_square(6,7), WHITE)) {
                add_move(out, from, make_square(6,7), 0, MF_CASTLE);
            }
        }
        if (b.castling_rights() & BQ) {
            if (!(b.pieces_all() & (bit(make_square(1,7)) | bit(make_square(2,7)) | bit(make_square(3,7))))
                && !b.is_square_attacked(make_square(4,7), WHITE)
                && !b.is_square_attacked(make_square(3,7), WHITE)
                && !b.is_square_attacked(make_square(2,7), WHITE)) {
                add_move(out, from, make_square(2,7), 0, MF_CASTLE);
            }
        }
    }
}

static void gen_sliders(const Board& b, std::vector<Move>& out, bool captures_only, PieceType pt) {
    Color stm = b.side_to_move();
    U64 ours = b.pieces(stm);
    U64 theirs = b.pieces(~stm);
    U64 occ = b.pieces_all();
    U64 pieces = b.pieces_pt(stm, pt);
    while (pieces) {
        Square from = pop_lsb(pieces);
        U64 att = 0;
        if (pt == BISHOP) att = bishop_attacks(from, occ);
        else if (pt == ROOK) att = rook_attacks(from, occ);
        else if (pt == QUEEN) att = queen_attacks(from, occ);

        U64 targets = captures_only ? (att & theirs) : (att & ~ours);
        U64 t = targets;
        while (t) {
            Square to = pop_lsb(t);
            int flags = (theirs & bit(to)) ? MF_CAPTURE : MF_NONE;
            add_move(out, from, to, 0, flags);
        }
    }
}

void gen_pseudo(const Board& b, std::vector<Move>& out, bool captures_only) {
    out.clear();
    gen_pawns(b, out, captures_only);
    gen_knights(b, out, captures_only);
    gen_sliders(b, out, captures_only, BISHOP);
    gen_sliders(b, out, captures_only, ROOK);
    gen_sliders(b, out, captures_only, QUEEN);
    gen_kings(b, out, captures_only);
}

} // namespace chronos
