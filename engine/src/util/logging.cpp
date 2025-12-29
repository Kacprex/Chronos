#include "chronos/util/logging.h"
#include "chronos/util/chrono_time.h"
#include <fstream>
#include <filesystem>

namespace chronos {

void JsonlLogger::set_path(const std::string& path) {
    path_ = path;
}

static void ensure_parent_dir(const std::string& path) {
    try {
        std::filesystem::path p(path);
        auto parent = p.parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    } catch (...) {
        // best-effort
    }
}

void JsonlLogger::append_line(const std::string& json_object_line) {
    if (path_.empty()) return;
    ensure_parent_dir(path_);
    std::ofstream out(path_, std::ios::app);
    if (!out) return;
    out << json_object_line << "\n";
}

std::string JsonlLogger::escape_json(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // skip control chars
                } else {
                    o += c;
                }
        }
    }
    return o;
}

// We use steady clock for engine timing; for logs we just need monotonic-ish ms.
// If you prefer real wall clock, swap implementation later.
std::uint64_t JsonlLogger::unix_ms() {
    return now_ms();
}

} // namespace chronos
