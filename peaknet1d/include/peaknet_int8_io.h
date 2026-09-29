#pragma once
// Auto-generated PeakNet full-INT8 I/O quantization.

static constexpr float PEAKNET_INPUT_SCALE = 4.999990389e-02f;
static constexpr int32_t PEAKNET_INPUT_ZERO_POINT = -16;

static constexpr float PEAKNET_OUTPUT_SCALE = 3.906250000e-03f;
static constexpr int32_t PEAKNET_OUTPUT_ZERO_POINT = -128;

// After conv06, call noodle_sigmoid(&tensor).
