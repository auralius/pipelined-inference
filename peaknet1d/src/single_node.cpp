/**
 * Single-MCU PeakNet1D / new Noodle full-INT8 baseline.
 *
 * Purpose:
 *   Run the complete PeakNet1D model on one STM32F411 using the same
 *   full-INT8 parameters as the distributed 5-MCU implementation.
 *
 * Host-facing protocol is intentionally kept identical to the original
 * single-MCU binary baseline:
 *
 *   MCU -> PC: b'R'                                     READY
 *   PC  -> MCU: "ECG" + 256 float32 little-endian       input
 *   MCU -> PC: "SCB" + uint16 length + 256 float32      scores
 *
 * Internally:
 *   Float32 host ECG -> INT8 quantization -> Conv1..Conv6 -> INT8 sigmoid
 *   -> Float32 dequantization only for the host-facing SCB packet.
 *
 * This keeps the Python benchmark unchanged while making the complete
 * Noodle inference path full-INT8.
 */

#include <Arduino.h>
#include <math.h>

#ifndef NOODLE_USE_INT8
#error "Build single_node.cpp with -D NOODLE_USE_INT8"
#endif

#ifndef NOODLE_MAX_K
#error "Build single_node.cpp with -D NOODLE_MAX_K=9"
#endif

#if NOODLE_MAX_K < 9
#error "PeakNet Conv1 requires NOODLE_MAX_K >= 9"
#endif

#include "noodle.h"

static_assert(sizeof(NoodleData) == 1,
              "PeakNet full-INT8 baseline requires 1-byte NoodleData");

#include "conv01_int8.h"
#include "conv02_int8.h"
#include "conv03_int8.h"
#include "conv04_int8.h"
#include "conv05_int8.h"
#include "conv06_int8.h"

#ifndef LINK_BAUD
#define LINK_BAUD 576000
#endif

#ifndef ENABLE_USB_DEBUG
#define ENABLE_USB_DEBUG 0
#endif

#ifndef ACTIVITY_LED_PIN
#define ACTIVITY_LED_PIN PC13
#endif

#ifndef ACTIVITY_LED_ACTIVE_LOW
#define ACTIVITY_LED_ACTIVE_LOW 1
#endif

static const uint16_t L = 256;
static const uint16_t MAX_CH = 16;
static const uint32_t MAX_TENSOR_N = (uint32_t)L * (uint32_t)MAX_CH;

/*
 * Two reusable Noodle tensors replace the old two Float32 malloc buffers.
 *
 * A starts as the ECG input and alternates with B:
 *
 *   A(1ch)  --Conv1--> B(8ch)
 *   B(8ch)  --Conv2--> A(16ch)
 *   A(16ch) --Conv3--> B(16ch)
 *   B(16ch) --Conv4--> A(16ch)
 *   A(16ch) --Conv5--> B(16ch)
 *   B(16ch) --Conv6--> A(1ch)
 *                              -> sigmoid(A)
 *
 * Both buffers are pre-grown to 16x256 INT8 elements during setup so the
 * arena does not need to grow as channel counts increase during inference.
 */
static NoodleTensor TENSOR_A;
static NoodleTensor TENSOR_B;

static ConvMem CONV1;
static ConvMem CONV2;
static ConvMem CONV3;
static ConvMem CONV4;
static ConvMem CONV5;
static ConvMem CONV6;

/* Original single-MCU host packet: "ECG" + 256 Float32 values. */
static uint8_t RX_RAW[3 + L * 4];


/* -------------------------------------------------------------------------- */
/* Activity LED                                                               */
/* -------------------------------------------------------------------------- */

static void activity_led_init() {
  pinMode(ACTIVITY_LED_PIN, OUTPUT);
#if ACTIVITY_LED_ACTIVE_LOW
  digitalWrite(ACTIVITY_LED_PIN, HIGH);
#else
  digitalWrite(ACTIVITY_LED_PIN, LOW);
#endif
}

static void activity_led_on() {
#if ACTIVITY_LED_ACTIVE_LOW
  digitalWrite(ACTIVITY_LED_PIN, LOW);
#else
  digitalWrite(ACTIVITY_LED_PIN, HIGH);
#endif
}

static void activity_led_off() {
#if ACTIVITY_LED_ACTIVE_LOW
  digitalWrite(ACTIVITY_LED_PIN, HIGH);
#else
  digitalWrite(ACTIVITY_LED_PIN, LOW);
#endif
}


/* -------------------------------------------------------------------------- */
/* Host protocol helpers                                                      */
/* -------------------------------------------------------------------------- */

static bool recv_exact(Stream &s, uint8_t *dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  const uint32_t t0 = millis();

  while (got < n) {
    if ((millis() - t0) > timeout_ms) return false;

    const int c = s.read();
    if (c < 0) {
      delay(1);
      continue;
    }

    dst[got++] = (uint8_t)c;
  }

  return true;
}

static float read_f32_le(const uint8_t *p) {
  float f;
  memcpy(&f, p, sizeof(f));
  return f;
}

static void write_u16_le(Stream &s, uint16_t v) {
  s.write((uint8_t)(v & 0xFF));
  s.write((uint8_t)((v >> 8) & 0xFF));
}

static void write_f32_le(Stream &s, float f) {
  uint8_t b[4];
  memcpy(b, &f, sizeof(f));
  s.write(b, sizeof(b));
}

static float sanitize_score(float x) {
  if (!isfinite(x)) return 0.0f;
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static void send_ready(Stream &s) {
  s.write((uint8_t)'R');
  s.flush();
}


/* -------------------------------------------------------------------------- */
/* Float32 host input -> INT8 Noodle input                                    */
/* -------------------------------------------------------------------------- */

static bool recv_ecg_frame_int8(Stream &s,
                                NoodleTensor *input,
                                uint32_t timeout_ms) {
  if (!recv_exact(s, RX_RAW, sizeof(RX_RAW), timeout_ms)) return false;

  if (RX_RAW[0] != 'E' || RX_RAW[1] != 'C' || RX_RAW[2] != 'G') {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR bad ECG header"));
#endif
    return false;
  }

  /*
   * require_1d() changes the logical shape to [1][256] but keeps the larger
   * retained buffer capacity allocated during setup.
   */
  NoodleData *x = noodle_tensor_require_1d(input, 1, L);
  if (!x) return false;

  noodle_tensor_set_quantization(
      input, CONV1.input_scale, CONV1.input_zero_point);

  const uint8_t *p = RX_RAW + 3;
  for (uint16_t i = 0; i < L; ++i, p += 4) {
    const float f = read_f32_le(p);
    x[i] = (NoodleData)noodle_quantize_float(
        f, CONV1.input_scale, CONV1.input_zero_point);
  }

  return true;
}


/* -------------------------------------------------------------------------- */
/* Complete full-INT8 PeakNet1D inference                                     */
/* -------------------------------------------------------------------------- */

static uint16_t predict_peaknet_int8() {
  uint16_t V = L;

  V = noodle_conv1d(&TENSOR_A, &TENSOR_B, CONV1);
  if (V != L) return 0;

  V = noodle_conv1d(&TENSOR_B, &TENSOR_A, CONV2);
  if (V != L) return 0;

  V = noodle_conv1d(&TENSOR_A, &TENSOR_B, CONV3);
  if (V != L) return 0;

  V = noodle_conv1d(&TENSOR_B, &TENSOR_A, CONV4);
  if (V != L) return 0;

  V = noodle_conv1d(&TENSOR_A, &TENSOR_B, CONV5);
  if (V != L) return 0;

  V = noodle_conv1d(&TENSOR_B, &TENSOR_A, CONV6);
  if (V != L) return 0;

  /*
   * New Noodle INT8 sigmoid converts the logits in-place to:
   *   scale      = 1/256
   *   zero point = -128
   */
  V = noodle_sigmoid(&TENSOR_A);
  if (V != L) return 0;

  return V;
}


/* -------------------------------------------------------------------------- */
/* INT8 sigmoid output -> legacy Float32 SCB packet                           */
/* -------------------------------------------------------------------------- */

static bool send_scores_binary(Stream &s,
                               const NoodleTensor *scores,
                               uint16_t length) {
  if (!scores || !noodle_tensor_const_data(scores)) return false;
  if (scores->rank != NOODLE_TENSOR_1D ||
      scores->C != 1 ||
      scores->W != length) {
    return false;
  }

  s.write((const uint8_t *)"SCB", 3);
  write_u16_le(s, length);

  const NoodleData *q = noodle_tensor_const_data(scores);

  for (uint16_t i = 0; i < length; ++i) {
    const float f = sanitize_score(
        noodle_dequantize_int8(
            (int8_t)q[i], scores->scale, scores->zero_point));

    write_f32_le(s, f);
  }

  s.flush();
  return true;
}


/* -------------------------------------------------------------------------- */
/* Noodle initialization                                                      */
/* -------------------------------------------------------------------------- */

static void init_noodle_int8() {
  CONV1 = peaknet_l01_make();
  CONV2 = peaknet_l02_make();
  CONV3 = peaknet_l03_make();
  CONV4 = peaknet_l04_make();
  CONV5 = peaknet_l05_make();
  CONV6 = peaknet_l06_make();

  noodle_tensor_init(&TENSOR_A);
  noodle_tensor_init(&TENSOR_B);

  /*
   * Pre-grow both reusable tensors to the largest PeakNet activation.
   * In INT8 this is only:
   *
   *     256 * 16 = 4096 bytes per tensor
   *
   * versus 16384 bytes per tensor in Float32.
   */
  if (!noodle_tensor_require_1d(&TENSOR_A, MAX_CH, L) ||
      !noodle_tensor_require_1d(&TENSOR_B, MAX_CH, L)) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERROR: Noodle INT8 tensor allocation failed"));
#endif
    while (true) delay(1000);
  }

  /*
   * Set the first tensor to the model input quantization. The logical [1][256]
   * shape is applied when the first ECG packet is received.
   */
  noodle_tensor_set_quantization(
      &TENSOR_A, CONV1.input_scale, CONV1.input_zero_point);
}


/* -------------------------------------------------------------------------- */
/* Arduino entry points                                                       */
/* -------------------------------------------------------------------------- */

void setup() {
  activity_led_init();

  Serial.begin(LINK_BAUD);

#if defined(USBCON) || defined(USBD_USE_CDC)
  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 2000) {
    delay(2);
  }
#endif

  init_noodle_int8();

  /*
   * Same request/response behavior as the Float32 single-MCU baseline:
   * advertise one input slot as available.
   */
  send_ready(Serial);
}

void loop() {
  if (!recv_ecg_frame_int8(Serial, &TENSOR_A, 10000)) {
    send_ready(Serial);
    return;
  }

  activity_led_on();

  const unsigned long t0 = micros();
  const uint16_t V = predict_peaknet_int8();
  const unsigned long infer_us = micros() - t0;

  if (V != L) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR full-INT8 PeakNet inference failed"));
#endif
    activity_led_off();
    send_ready(Serial);
    return;
  }

  if (!send_scores_binary(Serial, &TENSOR_A, V)) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR sending SCB scores"));
#endif
    activity_led_off();
    send_ready(Serial);
    return;
  }

#if ENABLE_USB_DEBUG
  /*
   * Do not enable this on the same Serial link during benchmark collection;
   * text debug would corrupt the binary protocol.
   */
  Serial.print(F("INT8 inference us="));
  Serial.println(infer_us);
#else
  (void)infer_us;
#endif

  activity_led_off();

  send_ready(Serial);
}
