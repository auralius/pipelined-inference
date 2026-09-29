#pragma once
#include <stdint.h>
#include <noodle.h>

#ifndef NOODLE_USE_INT8
#error "PeakNet INT8 headers require -D NOODLE_USE_INT8"
#endif

// Auto-generated from full-INT8 TFLite.
// Noodle Conv1D layout: [O][I][K].
// Layer 6: K=1, Cin=16, Cout=1

static const int8_t peaknet_l06_w[] = {
  -48, 13, -56, 43, -127, 82, -83, -95, -67, 42, -44, 48,
  -97, -68, -46, -17
};
static const int32_t peaknet_l06_b[] = {
  -67
};
static const int32_t peaknet_l06_mult[] = {
  1201292199
};
static const int32_t peaknet_l06_shift[] = {
  -6
};

static inline ConvMem peaknet_l06_make() {
  ConvMem c{};
  c.K = 1;
  c.P = 0;
  c.S = 1;
  c.O = 1;
  c.weight = peaknet_l06_w;
  c.bias = peaknet_l06_b;
  c.multiplier = peaknet_l06_mult;
  c.shift = peaknet_l06_shift;
  c.act = ACT_NONE;
  c.input_scale = 1.346284598e-01f;
  c.output_scale = 7.838127762e-02f;
  c.input_zero_point = -128;
  c.output_zero_point = 28;
  c.activation_min = -128;
  c.activation_max = 127;
  return c;
}
