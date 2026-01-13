#include "chronos/fen.h"
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace chronos {

static void init_empty(Fen& f) {
    for (int r=0;r<8;r++) for (int c=0;c<8;c++) f.board[r][c]='.';
}

Fen Fen::parse(const std::string& fen_str) {
    Fen f;
    init_empty(f);

    std::istringstream iss(fen_str);
    std::string placement, stm, castling, ep;
    if (!(iss >> placement >> stm >> castling >> ep)) {
        throw std::runtime_error("Invalid FEN: missing fields");
    }
    iss >> f.halfmove_clock >> f.fullmove_number;

    // placement
    int r = 0, c = 0;
    for (char ch : placement) {
        if (ch == '/') {
            r++; c = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            int n = ch - '0';
            for (int i=0;i<n;i++) {
                if (c>=8) throw std::runtime_error("Invalid FEN: file overflow");
                f.board[r][c++]='.';
            }
        } else {
            if (c>=8) throw std::runtime_error("Invalid FEN: file overflow");
            f.board[r][c++]=ch;
        }
    }
    if (r != 7) {
        // permissive but sanity check
    }

    // side to move
    if (stm == "w") f.white_to_move = true;
    else if (stm == "b") f.white_to_move = false;
    else throw std::runtime_error("Invalid FEN: side to move");

    // castling
    f.castle_wk = castling.find('K') != std::string::npos;
    f.castle_wq = castling.find('Q') != std::string::npos;
    f.castle_bk = castling.find('k') != std::string::npos;
    f.castle_bq = castling.find('q') != std::string::npos;

    return f;
}

} // namespace chronos
