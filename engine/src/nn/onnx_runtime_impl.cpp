#include "chronos/nn/nn_interface.h"
#include <string>

namespace chronos {

#if defined(CHRONOS_WITH_ONNX)

#include <array>
#include <memory>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include "chronos/nn/move_index.h"

#if defined(_WIN32)
#include <windows.h>
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return {};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring out(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), out.data(), wlen);
    return out;
}
#endif

struct ORTState {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "chronos"};
    Ort::SessionOptions so{};
    std::unique_ptr<Ort::Session> session{};
};

static ORTState* get_state(NNEvaluator::Impl* impl) {
    if (!impl) return nullptr;
    return reinterpret_cast<ORTState*>(impl->opaque);
}

bool chronos_onnx_load(NNEvaluator::Impl* impl, const NNConfig& cfg, std::string& err) {
    try {
        if (!impl) { err = "Impl is null"; return false; }

        auto st = std::make_unique<ORTState>();
        st->so.SetIntraOpNumThreads(cfg.intra_threads);
        st->so.SetInterOpNumThreads(cfg.inter_threads);
        st->so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // CUDA EP could be added here if you link onnxruntime-gpu + provider API.

#if defined(_WIN32)
        std::wstring wpath = utf8_to_wide(cfg.model_path);
        st->session = std::make_unique<Ort::Session>(st->env, wpath.c_str(), st->so);
#else
        st->session = std::make_unique<Ort::Session>(st->env, cfg.model_path.c_str(), st->so);
#endif

        delete get_state(impl);
        impl->opaque = st.release();
        return true;
    } catch (const Ort::Exception& e) {
        err = std::string("ORT load failed: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        err = std::string("ONNX load failed: ") + e.what();
        return false;
    }
}

bool chronos_onnx_eval(NNEvaluator::Impl* impl, const NNConfig& cfg, const NNEncoding& enc, NNOutput& out, std::string& err) {
    try {
        ORTState* st = get_state(impl);
        if (!st || !st->session) { err = "ORT session not initialized"; return false; }

        std::array<int64_t, 4> shape = {1, enc.planes, 8, 8};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(enc.data.data()),
            enc.data.size(), shape.data(), shape.size()
        );

        const char* in_name = cfg.input_name.c_str();
        const char* out_name = cfg.output_name.c_str();

        auto outputs = st->session->Run(
            Ort::RunOptions{nullptr},
            &in_name, &input, 1,
            &out_name, 1
        );

        if (outputs.empty()) { err = "ORT returned no outputs"; return false; }

        auto ti = outputs[0].GetTensorTypeAndShapeInfo();
        size_t n = (size_t)ti.GetElementCount();
        float* y = outputs[0].GetTensorMutableData<float>();

        if (n == 4) {
            out.value = y[0];
            out.pressure = y[1];
            out.volatility = y[2];
            out.complexity = y[3];
            out.has_policy = false;
            out.policy_logits.clear();
            return true;
        }

        // Standard: [value, pressure, volatility, complexity, policy_logits...]
        if (n == (size_t)(4 + MOVE_SPACE)) {
            out.value = y[0];
            out.pressure = y[1];
            out.volatility = y[2];
            out.complexity = y[3];
            out.has_policy = true;
            out.policy_logits.assign(y + 4, y + 4 + MOVE_SPACE);
            return true;
        }

        err = "Unexpected ONNX output size: " + std::to_string(n) + " (expected 4 or 4+MOVE_SPACE)";
        return false;
    } catch (const Ort::Exception& e) {
        err = std::string("ORT eval failed: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        err = std::string("ONNX eval failed: ") + e.what();
        return false;
    }
}

void chronos_onnx_free(NNEvaluator::Impl* impl) {
    if (!impl) return;
    auto st = get_state(impl);
    delete st;
    impl->opaque = nullptr;
}

#else // CHRONOS_WITH_ONNX

bool chronos_onnx_load(NNEvaluator::Impl*, const NNConfig&, std::string& err) {
    err = "Chronos built without ONNX Runtime.";
    return false;
}
bool chronos_onnx_eval(NNEvaluator::Impl*, const NNConfig&, const NNEncoding&, NNOutput&, std::string& err) {
    err = "Chronos built without ONNX Runtime.";
    return false;
}
void chronos_onnx_free(NNEvaluator::Impl*) {}

#endif

} // namespace chronos
