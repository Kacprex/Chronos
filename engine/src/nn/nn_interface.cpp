#include "chronos/nn/nn_interface.h"
#include "chronos/util/bitops.h"
#include <memory>

namespace chronos {

// Plane order (stable, 25):
// 0..5  White: P,N,B,R,Q,K
// 6..11 Black: p,n,b,r,q,k
// 12    side to move
// 13..16 castling KQkq
// 17..24 ep file planes (8)
static constexpr int PLANES = 12 + 1 + 4 + 8;

static int piece_plane(Piece p) {
    switch (p) {
        case W_PAWN: return 0;
        case W_KNIGHT: return 1;
        case W_BISHOP: return 2;
        case W_ROOK: return 3;
        case W_QUEEN: return 4;
        case W_KING: return 5;
        case B_PAWN: return 6;
        case B_KNIGHT: return 7;
        case B_BISHOP: return 8;
        case B_ROOK: return 9;
        case B_QUEEN: return 10;
        case B_KING: return 11;
        default: return -1;
    }
}

NNEncoding encode_position(const Board& b) {
    NNEncoding enc;
    enc.planes = PLANES;
    enc.data.assign(PLANES * 64, 0.0f);

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = b.piece_on(sq);
        int pl = piece_plane(p);
        if (pl >= 0) enc.data[pl * 64 + sq] = 1.0f;
    }

    float stm = (b.side_to_move() == WHITE) ? 1.0f : 0.0f;
    for (int sq = 0; sq < 64; ++sq) enc.data[12 * 64 + sq] = stm;

    auto fill_plane = [&](int plane, float v) {
        for (int sq = 0; sq < 64; ++sq) enc.data[plane * 64 + sq] = v;
    };

    fill_plane(13, (b.castling_rights() & WK) ? 1.0f : 0.0f);
    fill_plane(14, (b.castling_rights() & WQ) ? 1.0f : 0.0f);
    fill_plane(15, (b.castling_rights() & BK) ? 1.0f : 0.0f);
    fill_plane(16, (b.castling_rights() & BQ) ? 1.0f : 0.0f);

    for (int i = 0; i < 8; ++i) {
        float v = 0.0f;
        if (b.ep_square() != SQ_NONE && file_of(b.ep_square()) == i) v = 1.0f;
        fill_plane(17 + i, v);
    }

    return enc;
}

struct NNEvaluator::Impl {
#if defined(CHRONOS_WITH_ONNX)
    void* opaque = nullptr;
#endif
};

NNEvaluator::NNEvaluator() = default;

NNEvaluator::~NNEvaluator() {
#if defined(CHRONOS_WITH_ONNX)
    if (impl_) {
        extern void chronos_onnx_free(NNEvaluator::Impl* impl);
        chronos_onnx_free(impl_);
    }
#endif
    delete impl_;
    impl_ = nullptr;
}

bool NNEvaluator::load(const NNConfig& cfg) {
    cfg_ = cfg;
    enabled_ = cfg.enabled && !cfg.model_path.empty();
    backend_ready_ = false;
    last_error_.clear();

    if (!enabled_) return false;

#if defined(CHRONOS_WITH_ONNX)
    if (!impl_) impl_ = new Impl();

    extern bool chronos_onnx_load(NNEvaluator::Impl* impl, const NNConfig& cfg, std::string& err);
    backend_ready_ = chronos_onnx_load(impl_, cfg_, last_error_);
    return backend_ready_;
#else
    last_error_ = "Chronos was built without ONNX Runtime (CHRONOS_WITH_ONNX=OFF).";
    return false;
#endif
}

NNOutput NNEvaluator::eval(const Board& b) {
    NNOutput out{};
    if (!enabled_) return out;

#if defined(CHRONOS_WITH_ONNX)
    if (!backend_ready_ || !impl_) return out;

    NNEncoding enc = encode_position(b);
    extern bool chronos_onnx_eval(NNEvaluator::Impl* impl, const NNConfig& cfg, const NNEncoding& enc, NNOutput& out, std::string& err);

    if (!chronos_onnx_eval(impl_, cfg_, enc, out, last_error_)) return NNOutput{};
    return out;
#else
    (void)b;
    return out;
#endif
}

} // namespace chronos
