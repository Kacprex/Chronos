#include "chronos/search/tt.h"
#include <algorithm>

namespace chronos {

void TranspositionTable::resize_mb(int mb) {
    std::size_t bytes = std::size_t(mb) * 1024ULL * 1024ULL;
    std::size_t n = bytes / sizeof(TTEntry);
    if (n < 1) n = 1;

    // size to power of two
    std::size_t p2 = 1;
    while (p2 < n) p2 <<= 1;
    table_.assign(p2, TTEntry{});
    mask_ = p2 - 1;
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{});
}

bool TranspositionTable::probe(U64 key, TTEntry& out) const {
    if (table_.empty()) return false;
    const TTEntry& e = table_[key & mask_];
    if (e.key == key) { out = e; return true; }
    return false;
}

void TranspositionTable::store(U64 key, int depth, int score, TTFlag flag, Move best) {
    if (table_.empty()) return;
    TTEntry& e = table_[key & mask_];
    // replace if deeper or empty
    if (e.key != key || depth >= e.depth) {
        e.key = key;
        e.depth = depth;
        e.score = score;
        e.flag = flag;
        e.best = best;
    }
}

} // namespace chronos
