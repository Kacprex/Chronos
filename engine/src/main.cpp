#include "chronos/board/attacks.h"
#include "chronos/uci/uci.h"

int main() {
    chronos::init_attacks();
    chronos::UCI uci;
    uci.loop();
    return 0;
}
