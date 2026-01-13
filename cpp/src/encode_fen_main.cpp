#include <iostream>
#include <iomanip>
#include <string>
#include "chronos/fen.h"
#include "chronos/encoding.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: chronos_encode_fen \"<FEN>\"\n";
        return 1;
    }
    try {
        std::string fen_str = argv[1];
        auto f = chronos::Fen::parse(fen_str);
        auto x = chronos::encode_planes(f);

        // Print a small checksum + first 32 values so you can sanity check
        double sum = 0;
        for (float v : x) sum += v;

        std::cout << "INPUT_DIM=" << chronos::INPUT_DIM << "\n";
        std::cout << "SUM=" << std::fixed << std::setprecision(3) << sum << "\n";
        std::cout << "FIRST32:";
        for (int i=0;i<32;i++) std::cout << " " << x[i];
        std::cout << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
