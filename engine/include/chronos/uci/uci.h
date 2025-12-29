#pragma once
#include "../search/search.h"
#include "../board/board.h"
#include "../util/logging.h"
#include "../nn/nn_interface.h"
#include <string>

namespace chronos {

class UCI {
public:
    UCI();
    void loop();

private:
    void cmd_uci();
    void cmd_isready();
    void cmd_ucinewgame();
    void cmd_position(const std::string& line);
    void cmd_go(const std::string& line);
    void cmd_stop();
    void cmd_setoption(const std::string& line);

    Board board_{};
    Searcher searcher_{};

    int hash_mb_ = 256;

    // Phase 2: structured JSONL logging
    JsonlLogger logger_{};
    bool log_enabled_ = true;
    std::string log_path_{};
    std::string run_id_ = "uci";

    // Phase 3: NN hook (may be stub)
    NNConfig nn_cfg_{};
    NNEvaluator nn_{};

    // Phase 4: Hybrid arbitration knobs
    bool hybrid_enabled_ = true;
    std::string mode_ = "classic"; // "classic" | "blitz"
    int accept_worse_cp_ = 30;
    int topk_ = 6;
};

} // namespace chronos
