#pragma once
#include <string>
#include <cstdint>

namespace chronos {

// Minimal JSONL logger (Phase 2).
// Writes to E:/chronos/logs/events.jsonl by default (override via CHRONOS_DATA_ROOT or UCI option).
class JsonlLogger {
public:
    void set_path(const std::string& path);
    const std::string& path() const { return path_; }

    // Append one JSON line (must already be valid JSON object string, without trailing newline).
    void append_line(const std::string& json_object_line);

    // Convenience helpers
    static std::string escape_json(const std::string& s);
    static std::uint64_t unix_ms();

private:
    std::string path_;
};

} // namespace chronos
