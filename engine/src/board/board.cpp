#include "chronos/board/board.h"
#include "chronos/board/attacks.h"
#include "chronos/board/movegen.h"
#include <sstream>
#include <cctype>

namespace chronos {

static Piece piece_from_fen(char c) {
    switch (c) {
        case 'P': return W_PAWN; case 'N': return W_KNIGHT; case 'B': return W_BISHOP;
        case 'R': return W_ROOK; case 'Q': return W_QUEEN; case 'K': return W_KING;
        case 'p': return B_PAWN; case 'n': return B_KNIGHT; case 'b': return B_BISHOP;
        case 'r': return B_ROOK; case 'q': return B_QUEEN; case 'k': return B_KING;
        default:  return NO_PIECE;
    }
}

static PieceType pt_of(Piece p) {
    if (p == W_PAWN || p == B_PAWN) return PAWN;
    if (p == W_KNIGHT || p == B_KNIGHT) return KNIGHT;
    if (p == W_BISHOP || p == B_BISHOP) return BISHOP;
    if (p == W_ROOK || p == B_ROOK) return ROOK;
    if (p == W_QUEEN || p == B_QUEEN) return QUEEN;
    if (p == W_KING || p == B_KING) return KING;
    return PAWN;
}

Board::Board() { set_startpos(); }

void Board::clear() {
    mailbox_.fill(NO_PIECE);
    for (int c = 0; c < COLOR_NB; ++c) {
        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) bb_[c][pt] = 0;
        occ_color_[c] = 0;
        king_sq_[c] = SQ_NONE;
    }
    occ_all_ = 0;
    stm_ = WHITE;
    castling_ = 0;
    ep_ = SQ_NONE;
    halfmove_ = 0;
    fullmove_ = 1;
    history_.clear();
    hash_ = 0;
}

void Board::put_piece(Piece p, Square sq) {
    mailbox_[sq] = p;
    Color c = is_white(p) ? WHITE : BLACK;
    PieceType pt = pt_of(p);
    bb_[c][pt] |= bit(sq);
    occ_color_[c] |= bit(sq);
    occ_all_ |= bit(sq);
    if (pt == KING) king_sq_[c] = sq;

    hash_ ^= zobrist().piece_sq[p][sq];
}

void Board::remove_piece(Piece p, Square sq) {
    mailbox_[sq] = NO_PIECE;
    Color c = is_white(p) ? WHITE : BLACK;
    PieceType pt = pt_of(p);
    bb_[c][pt] &= ~bit(sq);
    occ_color_[c] &= ~bit(sq);
    occ_all_ &= ~bit(sq);

    hash_ ^= zobrist().piece_sq[p][sq];
}

void Board::move_piece(Piece p, Square from, Square to) {
    remove_piece(p, from);
    put_piece(p, to);
}

void Board::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Board::set_fen(const std::string& fen) {
    clear();

    std::istringstream iss(fen);
    std::string board, stm, cast, ep;
    int half = 0, full = 1;
    if (!(iss >> board >> stm >> cast >> ep >> half >> full)) return false;

    int sq = 56; // a8
    for (char c : board) {
        if (c == '/') { sq -= 16; continue; }
        if (std::isdigit((unsigned char)c)) { sq += (c - '0'); continue; }
        Piece p = piece_from_fen(c);
        if (p == NO_PIECE) return false;
        put_piece(p, sq);
        sq++;
    }

    stm_ = (stm == "w") ? WHITE : BLACK;
    if (stm_ == BLACK) hash_ ^= zobrist().side_to_move;

    castling_ = 0;
    if (cast.find('K') != std::string::npos) castling_ |= WK;
    if (cast.find('Q') != std::string::npos) castling_ |= WQ;
    if (cast.find('k') != std::string::npos) castling_ |= BK;
    if (cast.find('q') != std::string::npos) castling_ |= BQ;
    hash_ ^= zobrist().castling[castling_ & 15];

    if (ep != "-" ) {
        ep_ = string_to_square(ep);
    } else {
        ep_ = SQ_NONE;
    }
    int ep_file = (ep_ == SQ_NONE) ? 8 : file_of(ep_);
    hash_ ^= zobrist().ep_file[ep_file];

    halfmove_ = half;
    fullmove_ = full;

    return true;
}

void Board::apply_uci_moves(const std::vector<std::string>& moves) {
    for (const auto& mstr : moves) {
        Move m = uci_to_move(mstr, *this);
        // Ensure it's legal: we generate legal list and match
        std::vector<Move> legal;
        generate_legal(legal);
        bool ok = false;
        for (Move lm : legal) {
            if (move_to_uci(lm) == move_to_uci(m)) { // promotion flags alignment
                m = lm;
                ok = true;
                break;
            }
        }
        if (!ok) break;
        make(m);
    }
}

bool Board::is_square_attacked(Square sq, Color by) const {
    U64 occ = occ_all_;

    // pawns attack *towards* enemy
    if (by == WHITE) {
        // white pawns attack from sq-7/sq-9 to sq
        int f = file_of(sq), r = rank_of(sq);
        if (r > 0) {
            if (f > 0) {
                Square ps = sq - 9;
                if (ps >= 0 && mailbox_[ps] == W_PAWN) return true;
            }
            if (f < 7) {
                Square ps = sq - 7;
                if (ps >= 0 && mailbox_[ps] == W_PAWN) return true;
            }
        }
    } else {
        int f = file_of(sq), r = rank_of(sq);
        if (r < 7) {
            if (f > 0) {
                Square ps = sq + 7;
                if (ps < 64 && mailbox_[ps] == B_PAWN) return true;
            }
            if (f < 7) {
                Square ps = sq + 9;
                if (ps < 64 && mailbox_[ps] == B_PAWN) return true;
            }
        }
    }

    // knights
    U64 kn = knight_attacks(sq);
    U64 knights = bb_[by][KNIGHT];
    if (kn & knights) return true;

    // kings
    U64 kg = king_attacks(sq);
    U64 king = bb_[by][KING];
    if (kg & king) return true;

    // bishops/queens
    U64 diag = bishop_attacks(sq, occ);
    U64 bishops = bb_[by][BISHOP] | bb_[by][QUEEN];
    if (diag & bishops) return true;

    // rooks/queens
    U64 ortho = rook_attacks(sq, occ);
    U64 rooks = bb_[by][ROOK] | bb_[by][QUEEN];
    if (ortho & rooks) return true;

    return false;
}

bool Board::in_check(Color c) const {
    return is_square_attacked(king_sq_[c], ~c);
}

bool Board::make_impl(Move m, UndoState& u) {
    u.move = m;
    u.captured = NO_PIECE;
    u.castling_rights = castling_;
    u.ep_square = ep_;
    u.halfmove_clock = halfmove_;
    u.fullmove_number = fullmove_;
    u.hash = hash_;

    // remove castling hash, ep hash
    hash_ ^= zobrist().castling[castling_ & 15];
    int old_ep_file = (ep_ == SQ_NONE) ? 8 : file_of(ep_);
    hash_ ^= zobrist().ep_file[old_ep_file];

    Square from = move_from(m);
    Square to = move_to(m);
    int flags = move_flags(m);
    int promo = move_promo(m);

    Piece moving = mailbox_[from];
    if (moving == NO_PIECE) return false;

    // halfmove clock
    if (moving == W_PAWN || moving == B_PAWN || (flags & MF_CAPTURE)) halfmove_ = 0;
    else halfmove_++;

    // clear ep by default
    ep_ = SQ_NONE;

    // capture
    if (flags & MF_ENPASSANT) {
        Square cap_sq = (stm_ == WHITE) ? (to - 8) : (to + 8);
        Piece cap = mailbox_[cap_sq];
        u.captured = cap;
        remove_piece(cap, cap_sq);
    } else if (flags & MF_CAPTURE) {
        Piece cap = mailbox_[to];
        u.captured = cap;
        if (cap != NO_PIECE) remove_piece(cap, to);
    }

    // move
    remove_piece(moving, from);

    // promotions
    if (flags & MF_PROMOTION) {
        Piece newp = NO_PIECE;
        if (stm_ == WHITE) {
            if (promo == 1) newp = W_KNIGHT;
            else if (promo == 2) newp = W_BISHOP;
            else if (promo == 3) newp = W_ROOK;
            else newp = W_QUEEN;
        } else {
            if (promo == 1) newp = B_KNIGHT;
            else if (promo == 2) newp = B_BISHOP;
            else if (promo == 3) newp = B_ROOK;
            else newp = B_QUEEN;
        }
        put_piece(newp, to);
    } else {
        put_piece(moving, to);
    }

    // castling move rook
    if (flags & MF_CASTLE) {
        if (stm_ == WHITE) {
            if (to == make_square(6,0)) { // O-O
                Piece rook = mailbox_[make_square(7,0)];
                remove_piece(rook, make_square(7,0));
                put_piece(rook, make_square(5,0));
            } else if (to == make_square(2,0)) { // O-O-O
                Piece rook = mailbox_[make_square(0,0)];
                remove_piece(rook, make_square(0,0));
                put_piece(rook, make_square(3,0));
            }
        } else {
            if (to == make_square(6,7)) {
                Piece rook = mailbox_[make_square(7,7)];
                remove_piece(rook, make_square(7,7));
                put_piece(rook, make_square(5,7));
            } else if (to == make_square(2,7)) {
                Piece rook = mailbox_[make_square(0,7)];
                remove_piece(rook, make_square(0,7));
                put_piece(rook, make_square(3,7));
            }
        }
    }

    // set ep square on double push
    if (flags & MF_DOUBLE_PUSH) {
        ep_ = (stm_ == WHITE) ? (to - 8) : (to + 8);
    }

    // update castling rights due to king/rook moves/captures
    auto update_castle_rook = [&](Square sq) {
        if (sq == make_square(0,0)) castling_ &= ~WQ;
        if (sq == make_square(7,0)) castling_ &= ~WK;
        if (sq == make_square(0,7)) castling_ &= ~BQ;
        if (sq == make_square(7,7)) castling_ &= ~BK;
    };

    if (moving == W_KING) castling_ &= ~(WK | WQ);
    if (moving == B_KING) castling_ &= ~(BK | BQ);
    if (moving == W_ROOK || moving == B_ROOK) update_castle_rook(from);
    if (u.captured == W_ROOK || u.captured == B_ROOK) update_castle_rook(to);

    // apply new castling hash, new ep hash
    hash_ ^= zobrist().castling[castling_ & 15];
    int new_ep_file = (ep_ == SQ_NONE) ? 8 : file_of(ep_);
    hash_ ^= zobrist().ep_file[new_ep_file];

    // switch side
    stm_ = ~stm_;
    hash_ ^= zobrist().side_to_move;

    if (stm_ == WHITE) fullmove_++;

    return true;
}

bool Board::make(Move m) {
    UndoState u;
    if (!make_impl(m, u)) return false;

    // legality: side that just moved is ~stm_ now
    Color moved = ~stm_;
    if (in_check(moved)) {
        undo_impl(u);
        return false;
    }
    history_.push_back(u);
    return true;
}

void Board::undo_impl(const UndoState& u) {
    // Restore by replaying from stored state (simplest correct MVP)
    // We store full hash and clocks/castling/ep, but board pieces must revert.
    // So we undo by reversing move with knowledge of captured, promotion, castle, ep.
    Move m = u.move;
    Square from = move_from(m);
    Square to = move_to(m);
    int flags = move_flags(m);

    // revert side
    stm_ = ~stm_;
    if (stm_ == BLACK) fullmove_--;

    // clear current ep/castling hashes are not needed; we will overwrite hash_ at end.

    // move piece back
    Piece moved_piece = mailbox_[to];

    // undo castle rook move first (since king is on to)
    if (flags & MF_CASTLE) {
        if (stm_ == WHITE) {
            if (to == make_square(6,0)) { // O-O
                Piece rook = mailbox_[make_square(5,0)];
                remove_piece(rook, make_square(5,0));
                put_piece(rook, make_square(7,0));
            } else if (to == make_square(2,0)) {
                Piece rook = mailbox_[make_square(3,0)];
                remove_piece(rook, make_square(3,0));
                put_piece(rook, make_square(0,0));
            }
        } else {
            if (to == make_square(6,7)) {
                Piece rook = mailbox_[make_square(5,7)];
                remove_piece(rook, make_square(5,7));
                put_piece(rook, make_square(7,7));
            } else if (to == make_square(2,7)) {
                Piece rook = mailbox_[make_square(3,7)];
                remove_piece(rook, make_square(3,7));
                put_piece(rook, make_square(0,7));
            }
        }
    }

    // remove moved piece from 'to'
    remove_piece(moved_piece, to);

    // handle promotion revert
    if (flags & MF_PROMOTION) {
        moved_piece = (stm_ == WHITE) ? W_PAWN : B_PAWN;
    }

    // put piece back to from
    put_piece(moved_piece, from);

    // restore captured
    if (flags & MF_ENPASSANT) {
        Square cap_sq = (stm_ == WHITE) ? (to - 8) : (to + 8);
        if (u.captured != NO_PIECE) put_piece(u.captured, cap_sq);
    } else if (flags & MF_CAPTURE) {
        if (u.captured != NO_PIECE) put_piece(u.captured, to);
    }

    // restore state variables + hash
    castling_ = u.castling_rights;
    ep_ = u.ep_square;
    halfmove_ = u.halfmove_clock;
    fullmove_ = u.fullmove_number;
    hash_ = u.hash;
}

void Board::undo() {
    if (history_.empty()) return;
    UndoState u = history_.back();
    history_.pop_back();
    undo_impl(u);
}

void Board::generate_legal(std::vector<Move>& out) const {
    out.clear();
    std::vector<Move> pseudo;
    gen_pseudo(*this, pseudo, false);

    Board copy = *this;
    for (Move m : pseudo) {
        if (copy.make(m)) {
            out.push_back(m);
            copy.undo();
        } else {
            // ensure board restored if illegal
            // copy.undo() already inside make failure does undo; for safety we reset from original
            copy = *this;
        }
    }
}

void Board::generate_captures(std::vector<Move>& out) const {
    out.clear();
    std::vector<Move> pseudo;
    gen_pseudo(*this, pseudo, true);

    Board copy = *this;
    for (Move m : pseudo) {
        if (copy.make(m)) {
            out.push_back(m);
            copy.undo();
        } else {
            copy = *this;
        }
    }
}

} // namespace chronos
