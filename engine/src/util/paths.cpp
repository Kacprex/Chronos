#include "chronos/util/paths.h"
#include <cstdlib>

namespace chronos {

static std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + "/" + b;
}

Paths default_paths() {
    Paths p;
    p.data_root = "E:/chronos";
    p.shards_dir = join(p.data_root, "shards");
    p.shards_sl_dir = join(p.shards_dir, "sl");
    p.shards_rl_dir = join(p.shards_dir, "rl");
    p.runs_dir = join(p.data_root, "runs");
    p.logs_dir = join(p.data_root, "logs");
    p.models_dir = join(p.data_root, "models");
    return p;
}

Paths from_env_or_default() {
    Paths p = default_paths();
    if (const char* env = std::getenv("CHRONOS_DATA_ROOT")) {
        p.data_root = env;
        p.shards_dir = join(p.data_root, "shards");
        p.shards_sl_dir = join(p.shards_dir, "sl");
        p.shards_rl_dir = join(p.shards_dir, "rl");
        p.runs_dir = join(p.data_root, "runs");
        p.logs_dir = join(p.data_root, "logs");
        p.models_dir = join(p.data_root, "models");
    }
    return p;
}

} // namespace chronos
