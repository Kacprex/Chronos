#pragma once
#include "../board/board.h"
#include <string>
#include <vector>

namespace chronos {

struct NNOutput {
    float value = 0.0f;
    float pressure = 0.0f;
    float volatility = 0.0f;
    float complexity = 0.0f;

    // Phase 8+: optional policy head (AlphaZero-style 4672 logits)
    bool has_policy = false;
    std::vector<float> policy_logits;
};

struct NNConfig {
    bool enabled = false;
    std::string model_path;      // ONNX model path
    std::string input_name = "input";
    std::string output_name = "output";

    // Phase 6: runtime knobs
    int intra_threads = 1;
    int inter_threads = 1;

    // If you build with onnxruntime-gpu + CUDA EP, you can later enable this.
    bool prefer_cuda = false;
};

struct NNEncoding {
    int planes = 0;
    std::vector<float> data; // size = planes*64
};

NNEncoding encode_position(const Board& b);

class NNEvaluator {
public:
    struct Impl;
NNEvaluator();
    ~NNEvaluator();

    bool load(const NNConfig& cfg);
    NNOutput eval(const Board& b);

    bool enabled() const { return enabled_; }          // requested + has model path
    bool backend_ready() const { return backend_ready_; } // compiled + loaded
    bool active() const { return enabled_ && backend_ready_; }

    const std::string& last_error() const { return last_error_; }

private:
    bool enabled_ = false;
    bool backend_ready_ = false;
    std::string last_error_{};
    NNConfig cfg_{};
    Impl* impl_ = nullptr;
};

} // namespace chronos
