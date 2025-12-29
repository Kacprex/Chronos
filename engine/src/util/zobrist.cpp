#include "chronos/util/zobrist.h"
#include <random>

namespace chronos {

static Zobrist Z;

static U64 rand_u64(std::mt19937_64& rng) {
    std::uniform_int_distribution<U64> dist(0, ~0ULL);
    return dist(rng);
}

const Zobrist& zobrist() {
    static bool inited = false;
    if (!inited) {
        std::mt19937_64 rng(0xC0FFEE1234ULL);
        for (int p = 0; p < 13; ++p)
            for (int s = 0; s < 64; ++s)
                Z.piece_sq[p][s] = rand_u64(rng);

        for (int i = 0; i < 16; ++i) Z.castling[i] = rand_u64(rng);
        for (int i = 0; i < 9; ++i) Z.ep_file[i] = rand_u64(rng);
        Z.side_to_move = rand_u64(rng);
        inited = true;
    }
    return Z;
}

} // namespace chronos
