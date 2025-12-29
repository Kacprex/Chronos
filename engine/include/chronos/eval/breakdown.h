#pragma once
#include "../board/board.h"
#include <cstdint>

namespace chronos {

// Phase 2: evaluation decomposition.
// Keep it simple now: material + piece-square tables (PST) are separated.
// Placeholders exist for future terms.
struct EvalBreakdown {
    int material = 0;
    int pst = 0;
    int mobility = 0;
    int king_safety = 0;
    int pawn_structure = 0;
    int space = 0;
    int restriction = 0;

    int total() const {
        return material + pst + mobility + king_safety + pawn_structure + space + restriction;
    }
};

EvalBreakdown evaluate_breakdown(const Board& b);

} // namespace chronos
