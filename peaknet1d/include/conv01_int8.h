#pragma once
#include <stdint.h>
#include <noodle.h>

#ifndef NOODLE_USE_INT8
#error "PeakNet INT8 headers require -D NOODLE_USE_INT8"
#endif

// Auto-generated from full-INT8 TFLite.
// Noodle Conv1D layout: [O][I][K].
// Layer 1: K=9, Cin=1, Cout=8

static const int8_t peaknet_l01_w[] = {
  51, -85, 49, -101, -17, 58, -87, 127, 62, -9, -47, -22,
  3, -98, -127, 104, -26, 37, 74, 66, 47, 127, -89, -78,
  58, 52, -100, 127, 39, -24, -37, -1, -7, 88, -119, -28,
  -48, 43, 2, -67, -80, 1, -4, 87, 127, 98, -93, -32,
  30, 12, 46, 88, 127, -69, 32, -64, -127, 100, -25, 24,
  -56, -51, -49, -127, -49, -22, 2, 40, -83, 80, -97, 3
};
static const int32_t peaknet_l01_b[] = {
  2040, -516, -421, 1715, 204, -311, 399, 868
};
static const int32_t peaknet_l01_mult[] = {
  1393904791, 1554942442, 2065499969, 1089954583, 1808339074, 1912709878, 1835612518, 1303688220
};
static const int32_t peaknet_l01_shift[] = {
  -7, -7, -7, -7, -7, -7, -7, -7
};

static inline ConvMem peaknet_l01_make() {
  ConvMem c{};
  c.K = 9;
  c.P = 4;
  c.S = 1;
  c.O = 8;
  c.weight = peaknet_l01_w;
  c.bias = peaknet_l01_b;
  c.multiplier = peaknet_l01_mult;
  c.shift = peaknet_l01_shift;
  c.act = ACT_RELU;
  c.input_scale = 4.999990389e-02f;
  c.output_scale = 5.359189585e-02f;
  c.input_zero_point = -16;
  c.output_zero_point = -128;
  c.activation_min = -128;
  c.activation_max = 127;
  return c;
}
