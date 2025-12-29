#pragma once
#include "../board/board.h"
#include "tt.h"
#include <cstdint>
#include <atomic>
#include <string>
#include <vector>

namespace chronos {

struct SearchLimits {
    int depth = 8;
    std::uint64_t movetime_ms = 0; // (unused MVP)
    std::uint64_t hard_stop_ms = 0;
};

struct SearchInfo {
    int depth_reached = 0;
    int score_cp = 0;
    std::uint64_t nodes = 0;
    Move best = 0;
};

// Phase 4: root candidates from the last completed depth.
struct RootLine {
    Move move = 0;
    int score_cp = 0; // root score from side-to-move perspective
};

class Searcher {
public:
    Searcher();
    void set_hash_mb(int mb);
    void new_game();

    SearchInfo search(Board& b, const SearchLimits& lim);

    // Root candidates (sorted best-first) from the last finished search depth.
    const std::vector<RootLine>& last_root_lines() const { return last_root_lines_; }

    void stop();

private:
    SearchInfo root_search(Board& b, int depth);

    int alphabeta(Board& b, int depth, int alpha, int beta);
    int quiescence(Board& b, int alpha, int beta);

    bool should_stop() const;

    TranspositionTable tt_{};
    std::atomic<bool> stop_{false};
    std::uint64_t nodes_{0};
    std::uint64_t hard_stop_ms_{0};

    std::vector<RootLine> last_root_lines_{};
};

} // namespace chronos
