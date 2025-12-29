#pragma once
#include <string>

namespace chronos {

// Phase 0 convention:
// All persistent artifacts (including shards) live under E:/chronos by default.
// This helper centralizes those paths. In later phases, load from YAML and/or env.

struct Paths {
    std::string data_root;
    std::string shards_dir;
    std::string shards_sl_dir;
    std::string shards_rl_dir;
    std::string runs_dir;
    std::string logs_dir;
    std::string models_dir;
};

Paths default_paths();              // defaults to E:/chronos/*
Paths from_env_or_default();        // CHRONOS_DATA_ROOT overrides root if set

} // namespace chronos
