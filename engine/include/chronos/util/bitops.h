#pragma once
#include "types.h"

namespace chronos {

inline int pop_lsb(U64& bb) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    bb &= bb - 1;
    return int(idx);
#else
    int idx = __builtin_ctzll(bb);
    bb &= bb - 1;
    return idx;
#endif
}

inline int lsb_index(U64 bb) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return int(idx);
#else
    return __builtin_ctzll(bb);
#endif
}

inline int popcount(U64 bb) {
#if defined(_MSC_VER)
    return int(__popcnt64(bb));
#else
    return __builtin_popcountll(bb);
#endif
}

inline U64 bit(Square s) { return 1ULL << s; }

} // namespace chronos
