#pragma once
#include "../util/types.h"
#include "../util/bitops.h"
#include "../util/zobrist.h"
#include "move.h"
#include <vector>
#include <string>
#include <array>

namespace chronos {

struct UndoState {
    Move move{};
    Piece captured{NO_PIECE};
    int castling_rights{};
    Square ep_square{SQ_NONE};
    int halfmove_clock{};
    int fullmove_number{};
    U64 hash{};
};

class Board {
public:
    Board();

    // Setup
    void set_startpos();
    bool set_fen(const std::string& fen);

    // UCI position helper
    void apply_uci_moves(const std::vector<std::string>& moves);

    // State
    Color side_to_move() const { return stm_; }
    int castling_rights() const { return castling_; }
    Square ep_square() const { return ep_; }
    int halfmove_clock() const { return halfmove_; }
    int fullmove_number() const { return fullmove_; }
    U64 hash() const { return hash_; }

    // Pieces
    Piece piece_on(Square sq) const { return mailbox_[sq]; }
    U64 pieces(Color c) const { return occ_color_[c]; }
    U64 pieces_all() const { return occ_all_; }
    U64 pieces_pt(Color c, PieceType pt) const { return bb_[c][pt]; }
    Square king_square(Color c) const { return king_sq_[c]; }

    // Move handling
    bool make(Move m);
    void undo();

    // Checks
    bool in_check(Color c) const;
    bool is_square_attacked(Square sq, Color by) const;

    // Move generation
    void generate_legal(std::vector<Move>& out) const;
    void generate_captures(std::vector<Move>& out) const;

private:
    // Internal helpers
    void clear();
    void put_piece(Piece p, Square sq);
    void remove_piece(Piece p, Square sq);
    void move_piece(Piece p, Square from, Square to);

    bool make_impl(Move m, UndoState& u);
    void undo_impl(const UndoState& u);

    // State
    Color stm_{WHITE};
    int castling_{0};
    Square ep_{SQ_NONE};
    int halfmove_{0};
    int fullmove_{1};

    std::array<Piece, 64> mailbox_{};
    std::array<std::array<U64, PIECE_TYPE_NB>, COLOR_NB> bb_{}; // [color][pt]
    std::array<U64, COLOR_NB> occ_color_{};
    U64 occ_all_{0};

    std::array<Square, COLOR_NB> king_sq_{SQ_NONE, SQ_NONE};

    U64 hash_{0};
    std::vector<UndoState> history_{};

    friend Move uci_to_move(const std::string& uci, const Board& b);
};

} // namespace chronos
