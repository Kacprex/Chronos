#pragma once
#include <string>

namespace chronos {

struct Fen {
    // piece placement 8x8, ranks 8->1, files a->h
    // chars: 'P','N','B','R','Q','K','p','n','b','r','q','k','.' for empty
    char board[8][8]{};

    bool white_to_move = true;
    bool castle_wk = false;
    bool castle_wq = false;
    bool castle_bk = false;
    bool castle_bq = false;

    int halfmove_clock = 0;
    int fullmove_number = 1;

    static Fen parse(const std::string& fen_str);
};

} // namespace chronos
