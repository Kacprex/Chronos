#include "chronos/search/search.h"
#include "chronos/eval/eval.h"
#include "chronos/util/chrono_time.h"
#include <algorithm>
#include <iostream>

namespace chronos {

static constexpr int INF = 30000;
static constexpr int MATE = 29000;

Searcher::Searcher() {
    tt_.resize_mb(256);
}

void Searcher::set_hash_mb(int mb) {
    tt_.resize_mb(mb);
}

void Searcher::new_game() {
    tt_.clear();
}

void Searcher::stop() { stop_.store(true, std::memory_order_relaxed); }

bool Searcher::should_stop() const {
    if (stop_.load(std::memory_order_relaxed)) return true;
    if (hard_stop_ms_ != 0 && now_ms() >= hard_stop_ms_) return true;
    return false;
}

int Searcher::quiescence(Board& b, int alpha, int beta) {
    if (should_stop()) return alpha;

    nodes_++;
    int stand = evaluate(b);
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;

    std::vector<Move> caps;
    b.generate_captures(caps);

    for (Move m : caps) {
        if (!b.make(m)) continue;
        int score = -quiescence(b, -beta, -alpha);
        b.undo();

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
        if (should_stop()) break;
    }
    return alpha;
}

int Searcher::alphabeta(Board& b, int depth, int alpha, int beta) {
    if (should_stop()) return alpha;

    nodes_++;

    bool inCheck = b.in_check(b.side_to_move());
    if (depth <= 0) return quiescence(b, alpha, beta);

    // TT probe
    TTEntry te;
    if (tt_.probe(b.hash(), te) && te.depth >= depth) {
        if (te.flag == TT_EXACT) return te.score;
        if (te.flag == TT_ALPHA && te.score <= alpha) return te.score;
        if (te.flag == TT_BETA  && te.score >= beta)  return te.score;
    }

    std::vector<Move> moves;
    b.generate_legal(moves);

    if (moves.empty()) {
        if (inCheck) return -MATE + (64 - depth);
        return 0;
    }

    // Basic ordering: TT best first
    if (te.best) {
        auto it = std::find(moves.begin(), moves.end(), te.best);
        if (it != moves.end()) std::iter_swap(moves.begin(), it);
    }

    int bestScore = -INF;
    Move bestMove = 0;
    int alphaOrig = alpha;

    for (Move m : moves) {
        if (!b.make(m)) continue;
        int score = -alphabeta(b, depth - 1, -beta, -alpha);
        b.undo();

        if (should_stop()) break;

        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    TTFlag flag = TT_EXACT;
    if (bestScore <= alphaOrig) flag = TT_ALPHA;
    else if (bestScore >= beta) flag = TT_BETA;
    tt_.store(b.hash(), depth, bestScore, flag, bestMove);

    return bestScore;
}

SearchInfo Searcher::root_search(Board& b, int depth) {
    SearchInfo info{};
    info.depth_reached = depth;

    last_root_lines_.clear();

    std::vector<Move> moves;
    b.generate_legal(moves);

    if (moves.empty()) {
        info.best = 0;
        info.score_cp = b.in_check(b.side_to_move()) ? (-MATE + (64 - depth)) : 0;
        info.nodes = nodes_;
        return info;
    }

    // Root ordering: TT best first
    TTEntry te;
    if (tt_.probe(b.hash(), te) && te.best) {
        auto it = std::find(moves.begin(), moves.end(), te.best);
        if (it != moves.end()) std::iter_swap(moves.begin(), it);
    }

    int bestScore = -INF;
    Move bestMove = moves[0];

    int alpha = -INF, beta = INF;
    int alphaOrig = alpha;

    for (Move m : moves) {
        if (!b.make(m)) continue;
        int score = -alphabeta(b, depth - 1, -beta, -alpha);
        b.undo();

        if (should_stop()) break;

        last_root_lines_.push_back(RootLine{m, score});

        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    std::sort(last_root_lines_.begin(), last_root_lines_.end(),
              [](const RootLine& a, const RootLine& b){ return a.score_cp > b.score_cp; });

    TTFlag flag = TT_EXACT;
    if (bestScore <= alphaOrig) flag = TT_ALPHA;
    else if (bestScore >= beta) flag = TT_BETA;
    tt_.store(b.hash(), depth, bestScore, flag, bestMove);

    info.best = bestMove;
    info.score_cp = bestScore;
    info.nodes = nodes_;
    return info;
}

SearchInfo Searcher::search(Board& b, const SearchLimits& lim) {
    stop_.store(false, std::memory_order_relaxed);
    nodes_ = 0;
    hard_stop_ms_ = lim.hard_stop_ms;

    SearchInfo info{};
    int maxDepth = lim.depth;

    int bestScore = 0;
    Move bestMove = 0;

    for (int d = 1; d <= maxDepth; ++d) {
        SearchInfo di = root_search(b, d);
        if (should_stop()) break;

        bestScore = di.score_cp;
        bestMove = di.best;

        info.depth_reached = d;
        info.score_cp = bestScore;
        info.nodes = nodes_;
        info.best = bestMove;

        std::cout << "info depth " << d
                  << " score cp " << bestScore
                  << " nodes " << nodes_
                  << " pv " << (bestMove ? move_to_uci(bestMove) : "0000")
                  << "\n" << std::flush;

        if (should_stop()) break;
    }

    info.depth_reached = std::max(info.depth_reached, 1);
    info.score_cp = bestScore;
    info.nodes = nodes_;
    info.best = bestMove;

    return info;
}

} // namespace chronos
