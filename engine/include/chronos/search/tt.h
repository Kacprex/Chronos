#pragma once
#include "../util/types.h"
#include "../board/move.h"
#include <vector>

namespace chronos {

enum TTFlag : int { TT_EXACT=0, TT_ALPHA=1, TT_BETA=2 };

struct TTEntry {
    U64 key{};
    int depth{};
    int score{};
    TTFlag flag{TT_EXACT};
    Move best{};
};

class TranspositionTable {
public:
    void resize_mb(int mb);
    void clear();
    bool probe(U64 key, TTEntry& out) const;
    void store(U64 key, int depth, int score, TTFlag flag, Move best);

private:
    std::vector<TTEntry> table_{};
    std::size_t mask_{0};
};

} // namespace chronos
