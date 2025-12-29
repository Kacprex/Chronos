#pragma once
#include "../board/board.h"
#include "breakdown.h"

namespace chronos {

int evaluate(const Board& b);
EvalBreakdown evaluate_breakdown(const Board& b);

} // namespace chronos
