#pragma once
#include <array>
#include "fen.h"

namespace chronos {

// 18 planes * 8 * 8
constexpr int PLANES = 18;
constexpr int H = 8;
constexpr int W = 8;
constexpr int INPUT_DIM = PLANES * H * W;

// plane indices
// 0..5 white P,N,B,R,Q,K
// 6..11 black p,n,b,r,q,k
// 12 side to move
// 13..16 castling WK,WQ,BK,BQ
// 17 halfmove clock plane

std::array<float, INPUT_DIM> encode_planes(const Fen& f);

} // namespace chronos
