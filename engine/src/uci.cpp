#include "chronos/uci/uci.h"
#include "chronos/util/chrono_time.h"
#include "chronos/util/logging.h"
#include "chronos/util/paths.h"
#include "chronos/board/move.h"
#include "chronos/eval/eval.h"
#include "chronos/nn/nn_interface.h"
#include "chronos/nn/move_index.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

namespace chronos {

static std::vector<std::string> split(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string t;
    while (iss >> t) out.push_back(t);
    return out;
}

static std::string make_event_json(
    const std::string& type,
    const std::string& run_id,
    const std::string& payload_json
) {
    std::ostringstream o;
    o << "{"
      << "\"ts_ms\":" << JsonlLogger::unix_ms() << ","
      << "\"type\":\"" << JsonlLogger::escape_json(type) << "\","
      << "\"run_id\":\"" << JsonlLogger::escape_json(run_id) << "\"";
    if (!payload_json.empty()) {
        o << "," << payload_json;
    }
    o << "}";
    return o.str();
}

static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }
static float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

static float complexity_proxy(const Board& b) {
    std::vector<Move> moves;
    b.generate_legal(moves);
    return clampf(float(moves.size()) / 40.0f, 0.0f, 1.0f);
}

UCI::UCI() {
    Paths p = from_env_or_default();
    log_path_ = p.logs_dir + "/events.jsonl";
    logger_.set_path(log_path_);
    log_enabled_ = true;

    nn_cfg_.enabled = false;
    nn_cfg_.model_path = "";

    hybrid_enabled_ = true;
    mode_ = "classic";
    accept_worse_cp_ = 30;
    topk_ = 6;
}

void UCI::cmd_uci() {
    std::cout << "id name Chronos\n";
    std::cout << "id author Kacprex\n";
    std::cout << "option name Hash type spin default " << hash_mb_ << " min 16 max 4096\n";

    std::cout << "option name Log type check default " << (log_enabled_ ? "true" : "false") << "\n";
    std::cout << "option name LogPath type string default " << log_path_ << "\n";
    std::cout << "option name RunId type string default uci\n";

    std::cout << "option name UseNN type check default false\n";
    std::cout << "option name NNModel type string default \n";
    std::cout << "option name NNIntraThreads type spin default 1 min 1 max 16\n";
    std::cout << "option name NNInterThreads type spin default 1 min 1 max 16\n";
    std::cout << "option name NNPreferCuda type check default false\n";

    // Phase 4: hybrid arbitration knobs
    std::cout << "option name Hybrid type check default true\n";
    std::cout << "option name Mode type string default classic\n"; // classic|blitz
    std::cout << "option name AcceptWorseCp type spin default " << accept_worse_cp_ << " min 0 max 500\n";
    std::cout << "option name TopK type spin default " << topk_ << " min 1 max 20\n";

    std::cout << "uciok\n" << std::flush;
}

void UCI::cmd_isready() {
    std::cout << "readyok\n" << std::flush;
}

void UCI::cmd_ucinewgame() {
    board_.set_startpos();
    searcher_.new_game();
    run_id_ = "uci";
    if (log_enabled_) {
        logger_.append_line(make_event_json("ucinewgame", run_id_, "\"msg\":\"new game\""));
    }
}

void UCI::cmd_setoption(const std::string& line) {
    auto parts = split(line);
    auto itName = std::find(parts.begin(), parts.end(), "name");
    auto itValue = std::find(parts.begin(), parts.end(), "value");
    if (itName == parts.end()) return;

    std::string name;
    for (auto it = itName + 1; it != parts.end() && it != itValue; ++it) {
        if (!name.empty()) name += " ";
        name += *it;
    }

    std::string value;
    if (itValue != parts.end()) {
        for (auto it = itValue + 1; it != parts.end(); ++it) {
            if (!value.empty()) value += " ";
            value += *it;
        }
    }

    if (name == "Hash" && !value.empty()) {
        int mb = clampi(std::stoi(value), 16, 4096);
        hash_mb_ = mb;
        searcher_.set_hash_mb(hash_mb_);
        return;
    }
    if (name == "Log") {
        log_enabled_ = (value == "true" || value == "1" || value == "on");
        return;
    }
    if (name == "LogPath" && !value.empty()) {
        log_path_ = value;
        logger_.set_path(log_path_);
        return;
    }
    if (name == "RunId" && !value.empty()) {
        run_id_ = value;
        return;
    }
    if (name == "UseNN") {
        nn_cfg_.enabled = (value == "true" || value == "1" || value == "on");
        nn_.load(nn_cfg_);
        return;
    }
    if (name == "NNModel") {
        nn_cfg_.model_path = value;
        nn_.load(nn_cfg_);
        return;
    }
    if (name == "NNIntraThreads" && !value.empty()) {
        nn_cfg_.intra_threads = clampi(std::stoi(value), 1, 16);
        nn_.load(nn_cfg_);
        return;
    }
    if (name == "NNInterThreads" && !value.empty()) {
        nn_cfg_.inter_threads = clampi(std::stoi(value), 1, 16);
        nn_.load(nn_cfg_);
        return;
    }
    if (name == "NNPreferCuda") {
        nn_cfg_.prefer_cuda = (value == "true" || value == "1" || value == "on");
        nn_.load(nn_cfg_);
        return;
    }

    // Phase 4
    if (name == "Hybrid") {
        hybrid_enabled_ = (value == "true" || value == "1" || value == "on");
        return;
    }
    if (name == "Mode" && !value.empty()) {
        mode_ = value;
        return;
    }
    if (name == "AcceptWorseCp" && !value.empty()) {
        accept_worse_cp_ = clampi(std::stoi(value), 0, 500);
        return;
    }
    if (name == "TopK" && !value.empty()) {
        topk_ = clampi(std::stoi(value), 1, 20);
        return;
    }
}

void UCI::cmd_position(const std::string& line) {
    auto parts = split(line);
    if (parts.size() < 2) return;

    std::size_t idx = 1;
    if (parts[idx] == "startpos") {
        board_.set_startpos();
        idx++;
    } else if (parts[idx] == "fen") {
        idx++;
        if (idx + 5 >= parts.size()) return;
        std::string fen = parts[idx] + " " + parts[idx+1] + " " + parts[idx+2] + " " + parts[idx+3] + " " + parts[idx+4] + " " + parts[idx+5];
        board_.set_fen(fen);
        idx += 6;
    } else {
        return;
    }

    if (idx < parts.size() && parts[idx] == "moves") {
        idx++;
        std::vector<std::string> moves;
        for (; idx < parts.size(); ++idx) moves.push_back(parts[idx]);
        board_.apply_uci_moves(moves);
    }
}

static float hybrid_score(const std::string& mode, bool use_nn, const NNOutput& nn, float complexity, bool is_capture, float prior) {
    float wC = (mode == "blitz") ? 1.20f : 0.55f;
    float wP = 0.90f;
    float wV = 0.60f;
    float wX = 0.50f;
    float wU = (mode == "blitz") ? 0.45f : 0.20f; // policy prior weight
    float capPenalty = (mode == "blitz") ? 0.20f : 0.10f;

    float p = use_nn ? nn.pressure : 0.0f;
    float v = use_nn ? nn.volatility : 0.0f;
    float x = use_nn ? nn.complexity : 0.0f;

    float s = 0.0f;
    s += wC * complexity;
    s += wP * p;
    s -= wV * v;
    s += wX * x;
    if (use_nn) s += wU * prior;
    if (is_capture) s -= capPenalty;
    return s;
}

void UCI::cmd_go(const std::string& line) {
    auto parts = split(line);

    SearchLimits lim;
    lim.depth = 8;

    std::uint64_t now = now_ms();
    std::uint64_t movetime = 0;
    int depth = 0;
    std::uint64_t wtime=0,btime=0,winc=0,binc=0;
    int movestogo = 0;

    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "depth" && i+1 < parts.size()) depth = std::stoi(parts[++i]);
        else if (parts[i] == "movetime" && i+1 < parts.size()) movetime = (std::uint64_t)std::stoull(parts[++i]);
        else if (parts[i] == "wtime" && i+1 < parts.size()) wtime = (std::uint64_t)std::stoull(parts[++i]);
        else if (parts[i] == "btime" && i+1 < parts.size()) btime = (std::uint64_t)std::stoull(parts[++i]);
        else if (parts[i] == "winc" && i+1 < parts.size()) winc = (std::uint64_t)std::stoull(parts[++i]);
        else if (parts[i] == "binc" && i+1 < parts.size()) binc = (std::uint64_t)std::stoull(parts[++i]);
        else if (parts[i] == "movestogo" && i+1 < parts.size()) movestogo = std::stoi(parts[++i]);
    }

    if (depth > 0) lim.depth = std::max(1, depth);

    if (movetime > 0) {
        lim.hard_stop_ms = now + movetime;
    } else if (wtime > 0 || btime > 0) {
        std::uint64_t t = (board_.side_to_move() == WHITE) ? wtime : btime;
        std::uint64_t inc = (board_.side_to_move() == WHITE) ? winc : binc;
        int mtg = (movestogo > 0) ? movestogo : 30;
        std::uint64_t alloc = t / (std::uint64_t)mtg + (inc * 8) / 10;
        if (alloc < 25) alloc = 25;
        lim.hard_stop_ms = now + alloc;
    } else {
        lim.hard_stop_ms = 0;
    }

    EvalBreakdown bd_before = evaluate_breakdown(board_);

    NNOutput nn_before{};
    bool use_nn = nn_.active();
    if (use_nn) nn_before = nn_.eval(board_);

    // Phase 10: policy priors for candidate moves (softmax over candidate logits)
    auto policy_logit_for = [&](Move m) -> float {
        if (!use_nn || !nn_before.has_policy) return 0.0f;
        int idx = move_to_index(board_, m);
        if (idx < 0 || idx >= (int)nn_before.policy_logits.size()) return 0.0f;
        return nn_before.policy_logits[(std::size_t)idx];
    };

    SearchInfo si = searcher_.search(board_, lim);

    Move chosen = si.best;

    const auto& rootLines = searcher_.last_root_lines();

    struct Cand {
        Move move{};
        int score_cp = -100000;
        float complexity = 0.0f;
        NNOutput nn{};
        float prior = 0.0f; // policy prior from nn_before (softmax over candidates)
        float hscore = -1e9f;
        bool is_capture = false;
    };

std::vector<Cand> cands;

    if (!rootLines.empty()) {
        int best_cp = rootLines.front().score_cp;
        int min_cp = best_cp - accept_worse_cp_;

        int take = std::min<int>(topk_, (int)rootLines.size());
        for (int i = 0; i < take; ++i) {
            const auto& rl = rootLines[i];
            if (rl.score_cp < min_cp) continue;

            Board tmp = board_;
            if (!tmp.make(rl.move)) continue;

            float comp = complexity_proxy(tmp); // opponent to move
            NNOutput nn{};
            if (use_nn) nn = nn_.eval(tmp);

            bool iscap = (move_flags(rl.move) & MF_CAPTURE) != 0;
            float raw_prior = policy_logit_for(rl.move);
            float hs = hybrid_score(mode_, use_nn, nn, comp, iscap, 0.0f /*filled after softmax*/);

            cands.push_back(Cand{rl.move, rl.score_cp, comp, nn, raw_prior, hs, iscap});
        }

        if (hybrid_enabled_ && !cands.empty()) {
            // convert raw policy logits in c.prior into softmax priors across candidates
            if (use_nn && nn_before.has_policy) {
                float mx = cands[0].prior;
                for (const auto& c : cands) if (c.prior > mx) mx = c.prior;
                double sum = 0.0;
                for (auto& c : cands) { c.prior = (float)std::exp((double)(c.prior - mx)); sum += c.prior; }
                if (sum > 0.0) {
                    for (auto& c : cands) c.prior = (float)((double)c.prior / sum);
                }
                // recompute hybrid score with priors
                for (auto& c : cands) {
                    c.hscore = hybrid_score(mode_, use_nn, c.nn, c.complexity, c.is_capture, c.prior);
                }
            } else {
                for (auto& c : cands) c.prior = 0.0f;
                for (auto& c : cands) c.hscore = hybrid_score(mode_, use_nn, c.nn, c.complexity, c.is_capture, c.prior);
            }

            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){
                if (a.hscore != b.hscore) return a.hscore > b.hscore;
                return a.score_cp > b.score_cp;
            });
            chosen = cands.front().move;
        }
    }

    std::string bestUci = chosen ? move_to_uci(chosen) : "0000";

    EvalBreakdown bd_after{};
    NNOutput nn_after{};
    bool after_ok = false;
    if (chosen && bestUci != "0000") {
        Board tmp = board_;
        if (tmp.make(chosen)) {
            bd_after = evaluate_breakdown(tmp);
            if (use_nn) nn_after = nn_.eval(tmp);
            after_ok = true;
        }
    }

    if (log_enabled_) {
        std::ostringstream payload;
        payload
          << "\"bestmove\":\"" << JsonlLogger::escape_json(bestUci) << "\","
          << "\"depth\":" << si.depth_reached << ","
          << "\"nodes\":" << si.nodes << ","
          << "\"score_cp\":" << si.score_cp << ","
          << "\"hybrid\":{"
             << "\"enabled\":" << (hybrid_enabled_ ? "true" : "false") << ","
             << "\"mode\":\"" << JsonlLogger::escape_json(mode_) << "\","
             << "\"accept_worse_cp\":" << accept_worse_cp_ << ","
             << "\"topk\":" << topk_ << ","
             << "\"use_nn\":" << (use_nn ? "true" : "false") << ","
             << "\"backend_ready\":" << (nn_.backend_ready() ? "true" : "false") << ","
             << "\"nn_error\":\"" << JsonlLogger::escape_json(nn_.last_error()) << "\""
          << "},"
          << "\"eval_before\":{"
            << "\"material\":" << bd_before.material << ","
            << "\"pst\":" << bd_before.pst << ","
            << "\"mobility\":" << bd_before.mobility << ","
            << "\"king_safety\":" << bd_before.king_safety << ","
            << "\"pawn_structure\":" << bd_before.pawn_structure << ","
            << "\"space\":" << bd_before.space << ","
            << "\"restriction\":" << bd_before.restriction << ","
            << "\"total\":" << bd_before.total()
          << "},"
          << "\"nn_before\":{"
            << "\"value\":" << nn_before.value << ","
            << "\"pressure\":" << nn_before.pressure << ","
            << "\"volatility\":" << nn_before.volatility << ","
            << "\"complexity\":" << nn_before.complexity
          << "}";

        if (!cands.empty()) {
            payload << ",\"candidates\":[";
            for (std::size_t i = 0; i < cands.size(); ++i) {
                const auto& c = cands[i];
                if (i) payload << ",";
                payload << "{"
                        << "\"m\":\"" << JsonlLogger::escape_json(move_to_uci(c.move)) << "\","
                        << "\"score_cp\":" << c.score_cp << ","
                        << "\"complexity\":" << c.complexity << ","
                        << "\"capture\":" << (c.is_capture ? "true" : "false") << ","
                        << "\"hscore\":" << c.hscore << ","
                        << "\"nn\":{"
                           << "\"value\":" << c.nn.value << ","
                           << "\"pressure\":" << c.nn.pressure << ","
                           << "\"volatility\":" << c.nn.volatility << ","
                           << "\"complexity\":" << c.nn.complexity
                        << "}"
                        << "}";
            }
            payload << "]";
        }

        if (after_ok) {
            payload
              << ",\"eval_after\":{"
                << "\"material\":" << bd_after.material << ","
                << "\"pst\":" << bd_after.pst << ","
                << "\"mobility\":" << bd_after.mobility << ","
                << "\"king_safety\":" << bd_after.king_safety << ","
                << "\"pawn_structure\":" << bd_after.pawn_structure << ","
                << "\"space\":" << bd_after.space << ","
                << "\"restriction\":" << bd_after.restriction << ","
                << "\"total\":" << bd_after.total()
              << "},"
              << "\"nn_after\":{"
                << "\"value\":" << nn_after.value << ","
                << "\"pressure\":" << nn_after.pressure << ","
                << "\"volatility\":" << nn_after.volatility << ","
                << "\"complexity\":" << nn_after.complexity
              << "}";
        }

        logger_.append_line(make_event_json("engine_move", run_id_, payload.str()));
    }

    std::cout << "bestmove " << bestUci << "\n" << std::flush;
}

void UCI::cmd_stop() {
    searcher_.stop();
}

void UCI::loop() {
    board_.set_startpos();
    if (log_enabled_) {
        logger_.append_line(make_event_json("engine_start", run_id_, "\"msg\":\"Chronos engine started\""));
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "uci") cmd_uci();
        else if (line == "isready") cmd_isready();
        else if (line == "ucinewgame") cmd_ucinewgame();
        else if (line.rfind("setoption", 0) == 0) cmd_setoption(line);
        else if (line.rfind("position", 0) == 0) cmd_position(line);
        else if (line.rfind("go", 0) == 0) cmd_go(line);
        else if (line == "stop") cmd_stop();
        else if (line == "quit") break;
    }
}

} // namespace chronos
