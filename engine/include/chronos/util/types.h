#pragma once
#include <cstdint>
#include <array>
#include <string>

namespace chronos {

using U64 = std::uint64_t;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
inline Color operator~(Color c) { return c == WHITE ? BLACK : WHITE; }

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 7, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
};

inline bool is_white(Piece p) { return p >= W_PAWN && p <= W_KING; }
inline bool is_black(Piece p) { return p >= B_PAWN && p <= B_KING; }

enum PieceType : int { PAWN=0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB };

enum CastlingRight : int {
    WK = 1 << 0,
    WQ = 1 << 1,
    BK = 1 << 2,
    BQ = 1 << 3
};

using Square = int; // 0..63 (a1=0, h8=63)
constexpr Square SQ_NONE = -1;

inline int file_of(Square s) { return s & 7; }
inline int rank_of(Square s) { return s >> 3; }
inline Square make_square(int file, int rank) { return (rank << 3) | file; }

inline std::string square_to_string(Square s) {
    if (s < 0 || s > 63) return "-";
    char f = char('a' + file_of(s));
    char r = char('1' + rank_of(s));
    return std::string() + f + r;
}

inline Square string_to_square(const std::string& str) {
    if (str.size() < 2) return SQ_NONE;
    char f = str[0], r = str[1];
    if (f < 'a' || f > 'h' || r < '1' || r > '8') return SQ_NONE;
    return make_square(f - 'a', r - '1');
}

} // namespace chronos
