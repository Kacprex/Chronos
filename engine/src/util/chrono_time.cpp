#include "chronos/util/chrono_time.h"
#include <chrono>

namespace chronos {

std::uint64_t now_ms() {
    using namespace std::chrono;
    return (std::uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

} // namespace chronos
