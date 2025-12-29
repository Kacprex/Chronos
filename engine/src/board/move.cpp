#include "chronos/board/move.h"
#include "chronos/board/board.h"
#include <cctype>

namespace chronos {

std::string move_to_uci(Move m) {
    std::string s = square_to_string(move_from(m)) + square_to_string(move_to(m));
    if (move_flags(m) & MF_PROMOTION) {
        char pc = 'q';
        switch (move_promo(m)) {
            case 1: pc = 'n'; break;
            case 2: pc = 'b'; break;
            case 3: pc = 'r'; break;
            case 4: pc = 'q'; break;
            default: pc = 'q'; break;
        }
        s.push_back(pc);
    }
    return s;
}

static int promo_from_char(char c) {
    c = char(std::tolower((unsigned char)c));
    if (c == 'n') return 1;
    if (c == 'b') return 2;
    if (c == 'r') return 3;
    if (c == 'q') return 4;
    return 0;
}

Move uci_to_move(const std::string& uci, const Board& b) {
    if (uci.size() < 4) return 0;
    Square from = string_to_square(uci.substr(0,2));
    Square to   = string_to_square(uci.substr(2,2));
    if (from == SQ_NONE || to == SQ_NONE) return 0;

    int promo = 0;
    int flags = MF_NONE;
    if (uci.size() >= 5) {
        promo = promo_from_char(uci[4]);
        if (promo) flags |= MF_PROMOTION;
    }

    Piece moving = b.piece_on(from);
    Piece captured = b.piece_on(to);
    if (captured != NO_PIECE) flags |= MF_CAPTURE;

    // detect ep
    if ((moving == W_PAWN || moving == B_PAWN) && to == b.ep_square() && b.piece_on(to) == NO_PIECE) {
        flags |= MF_ENPASSANT | MF_CAPTURE;
    }
    // detect castle
    if ((moving == W_KING || moving == B_KING) && (from == make_square(4,0) || from == make_square(4,7))) {
        if (to == make_square(6,0) || to == make_square(2,0) || to == make_square(6,7) || to == make_square(2,7))
            flags |= MF_CASTLE;
    }
    return make_move(from, to, promo, flags);
}

} // namespace chronos
