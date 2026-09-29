/**
 * Node5 HAL USART1 RX-DMA prefetch -- new Noodle full-INT8 Conv5/Conv6.
 *
 * Forward data path:
 *   USART1 RX = PA10, USART1 TX = PA9, pure HAL
 * ACK path:
 *   Ack RX = PA3, Ack TX = PA2, HardwareSerial
 *
 * Node5 special behavior:
 *   - Receives INT8 Conv4 tensor (TNI) from Node4 using USART1 RX DMA.
 *   - Computes Conv5 + Conv6 + sigmoid with the new NoodleTensor INT8 API.
 *   - Arms RX DMA for the next Conv4 tensor BEFORE sending READY to Node4.
 *   - Sends READY upstream to Node4.
 *   - Dequantizes the final INT8 probabilities and sends legacy Float32 SCB scores to the PC.
 *   - Waits for host score ACK byte 'A' on Ack RX = PA3.
 *
 * Important rule:
 *   READY is sent only after the next RX DMA buffer is armed.
 *
 * Wiring:
 *   Node4 PA9  -> Node5 PA10 : Conv4 tensor input
 *   Node5 PA9  -> FTDI #2 RXD: binary SCB score output
 *   Node5 PA2  -> Node4 PA3 : READY upstream to Node4
 *   FTDI #2 TXD -> Node5 PA3: host score ACK byte 'A'
 *   All GND common
 */

#include <Arduino.h>
#include "stm32f4xx_hal.h"
#ifndef NOODLE_USE_INT8
#error "Build the complete project with -D NOODLE_USE_INT8"
#endif
#ifndef NOODLE_MAX_K
#error "Build the complete project with -D NOODLE_MAX_K=9"
#endif
#if NOODLE_MAX_K < 9
#error "PeakNet Conv1 requires NOODLE_MAX_K >= 9"
#endif
#include "noodle.h"

static_assert(sizeof(NoodleData) == 1, "PeakNet INT8 nodes require 1-byte NoodleData");

#include "conv05_int8.h"
#include "conv06_int8.h"

static const uint16_t L = 256;
static const uint16_t IN_CH = 16;
static const uint16_t MID_CH = 16;
static const uint16_t OUT_CH = 1;

#ifndef LINK_BAUD
#define LINK_BAUD 576000
#endif

#ifndef NODE_ID
#define NODE_ID 5
#endif

#ifndef ENABLE_USB_DEBUG
#define ENABLE_USB_DEBUG 0
#endif

#ifndef USE_INTERMCU_ACK
#define USE_INTERMCU_ACK 1
#endif

#ifndef ACK_BAUD
#define ACK_BAUD LINK_BAUD
#endif

// --------------------------------------------------
// GPIO READY output to Node4
// --------------------------------------------------
// Node5 PB0 -> Node4 PB1
// All GND common.
#ifndef USE_GPIO_READY
#define USE_GPIO_READY 1
#endif

#ifndef READY_OUT_UPSTREAM_PIN
#define READY_OUT_UPSTREAM_PIN PB0
#endif

#ifndef READY_ACTIVE_LEVEL
#define READY_ACTIVE_LEVEL HIGH
#endif


#ifndef USE_HOST_SCORE_ACK
#define USE_HOST_SCORE_ACK 1
#endif

#ifndef HOST_SCORE_ACK_BYTE
#define HOST_SCORE_ACK_BYTE 'A'
#endif

// --------------------------------------------------
// Recovery configuration
// --------------------------------------------------
#ifndef NODE5_RX_TIMEOUT_MS
#define NODE5_RX_TIMEOUT_MS 10000
#endif

#ifndef NODE5_HOST_ACK_TIMEOUT_MS
#define NODE5_HOST_ACK_TIMEOUT_MS 3000
#endif

#ifndef NODE5_MAX_RECOVERY_EVENTS
#define NODE5_MAX_RECOVERY_EVENTS 8
#endif

// --------------------------------------------------
// Buffers
// --------------------------------------------------
// Node5:
//   TENSOR_IN  : input Conv4 tensor, [16][256]; later reused for final [1][256] score
//   TENSOR_TMP : Conv5 output, [16][256]
static NoodleTensor TENSOR_IN;   // Conv4 input; later reused for Conv6/sigmoid output
static NoodleTensor TENSOR_TMP;  // Conv5 output
static ConvMem CONV5;
static ConvMem CONV6;

// TNI packet: "TNI" + frame_id + shape + signed INT8 payload.
static uint8_t RX_RAW_NEXT[3 + 4 + 2 + 2 + L * IN_CH];

// --------------------------------------------------
// Pure HAL USART1 + DMA handles for forward data path
// USART1 pins on STM32F411 BlackPill:
//   PA9  = USART1_TX
//   PA10 = USART1_RX
//
// USART1_RX DMA mapping on STM32F411:
//   DMA2 Stream2 Channel4
// --------------------------------------------------
static UART_HandleTypeDef huart1_link;
static DMA_HandleTypeDef hdma_usart1_rx;

// --------------------------------------------------
// Host ACK UART remains HardwareSerial on PA3 RX.
// Inter-MCU READY to Node4 is now GPIO PB0.
// PA2 TX is unused by the GPIO READY path.
// --------------------------------------------------
#if USE_INTERMCU_ACK
HardwareSerial Ack(PA3, PA2);
#endif

// --------------------------------------------------
// On-board activity LED
// --------------------------------------------------
#ifndef ACTIVITY_LED_PIN
#define ACTIVITY_LED_PIN PC13
#endif

#ifndef ACTIVITY_LED_ACTIVE_LOW
#define ACTIVITY_LED_ACTIVE_LOW 1
#endif

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

// --------------------------------------------------
// ACK / READY helpers
// --------------------------------------------------
// --------------------------------------------------
// GPIO READY helper
// --------------------------------------------------
static void ready_gpio_init() {
#if USE_GPIO_READY
  pinMode(READY_OUT_UPSTREAM_PIN, OUTPUT);
  digitalWrite(READY_OUT_UPSTREAM_PIN, (READY_ACTIVE_LEVEL == HIGH) ? LOW : HIGH);
#endif
}

static void ready_set_upstream(bool ready) {
#if USE_GPIO_READY
  const uint8_t active = READY_ACTIVE_LEVEL;
  const uint8_t inactive = (READY_ACTIVE_LEVEL == HIGH) ? LOW : HIGH;
  digitalWrite(READY_OUT_UPSTREAM_PIN, ready ? active : inactive);
#else
  (void)ready;
#endif
}

static void ack_send_ready_upstream() {
#if USE_INTERMCU_ACK
  // READY means Node5 has already armed RX DMA and can accept the next Conv4 tensor.
  ready_set_upstream(true);
#endif
}

static bool wait_host_score_ack(uint32_t timeout_ms) {
#if USE_HOST_SCORE_ACK && USE_INTERMCU_ACK
  uint32_t t0 = millis();

  while ((millis() - t0) <= timeout_ms) {
    int c = Ack.read();
    if (c == HOST_SCORE_ACK_BYTE) {
      return true;
    }
    delay(1);
  }

#if ENABLE_USB_DEBUG
  Serial.println(F("ERR host score ACK timeout"));
#endif
  return false;
#else
  (void)timeout_ms;
  return true;
#endif
}

// --------------------------------------------------
// Serialization helpers
// --------------------------------------------------
static uint16_t read_u16_le(const uint8_t *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *p) {
  return ((uint32_t)p[0]) |
         ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static bool link_tx_bytes(const uint8_t *p, uint16_t n, uint32_t timeout_ms = 1000) {
  return HAL_UART_Transmit(&huart1_link, (uint8_t*)p, n, timeout_ms) == HAL_OK;
}

static bool link_tx_u16_le(uint16_t v) {
  uint8_t b[2] = {
    (uint8_t)(v & 0xFF),
    (uint8_t)((v >> 8) & 0xFF),
  };
  return link_tx_bytes(b, 2);
}

static bool link_tx_u32_le(uint32_t v) {
  uint8_t b[4] = {
    (uint8_t)(v & 0xFF),
    (uint8_t)((v >> 8) & 0xFF),
    (uint8_t)((v >> 16) & 0xFF),
    (uint8_t)((v >> 24) & 0xFF),
  };
  return link_tx_bytes(b, 4);
}

static bool link_tx_f32_le(float f) {
  uint8_t b[4];
  memcpy(b, &f, 4);
  return link_tx_bytes(b, 4);
}

static float sanitize_score(float x) {
  if (!isfinite(x)) return 0.0f;
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static bool decode_tensor_raw(const uint8_t *raw,
                              NoodleTensor *tensor,
                              uint16_t expected_length,
                              uint16_t expected_channels,
                              float input_scale, int32_t input_zero_point,
                              uint32_t *frame_id) {
  if (raw[0] != 'T' || raw[1] != 'N' || raw[2] != 'I') {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR INT8 tensor header (expected TNI)"));
#endif
    return false;
  }

  if (frame_id) *frame_id = read_u32_le(raw + 3);
  const uint16_t length = read_u16_le(raw + 7);
  const uint16_t channels = read_u16_le(raw + 9);
  if (length != expected_length || channels != expected_channels) {
#if ENABLE_USB_DEBUG
    Serial.print(F("ERR tensor shape L=")); Serial.print(length);
    Serial.print(F(" C=")); Serial.println(channels);
#endif
    return false;
  }

  NoodleData *dst = noodle_tensor_require_1d(tensor, channels, length);
  if (!dst) return false;
  noodle_tensor_set_quantization(tensor, input_scale, input_zero_point);
  const uint32_t n = (uint32_t)length * (uint32_t)channels;
  memcpy(dst, raw + 11, n);
  return true;
}

static bool send_scores_binary_hal(const NoodleTensor *scores,
                                   uint16_t length, uint32_t frame_id) {
  if (!scores || !noodle_tensor_const_data(scores)) return false;
  if (scores->rank != NOODLE_TENSOR_1D || scores->C != 1 || scores->W != length) return false;

  // Keep the host-facing SCB packet Float32 so the existing Python benchmark
  // remains unchanged. Only the MCU-to-MCU links use TNI/INT8.
  const uint8_t hdr[3] = {'S', 'C', 'B'};
  if (!link_tx_bytes(hdr, 3)) return false;
  if (!link_tx_u32_le(frame_id)) return false;
  if (!link_tx_u16_le(length)) return false;

  const NoodleData *q = noodle_tensor_const_data(scores);
  for (uint16_t i = 0; i < length; ++i) {
    const float f = sanitize_score(
        noodle_dequantize_int8((int8_t)q[i], scores->scale, scores->zero_point));
    if (!link_tx_f32_le(f)) return false;
  }
  return true;
}

static void link_hal_uart1_init() {
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_USART1_CLK_ENABLE();

  GPIO_InitTypeDef gpio;
  memset(&gpio, 0, sizeof(gpio));
  gpio.Pin = GPIO_PIN_9 | GPIO_PIN_10;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_PULLUP;
  gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  gpio.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOA, &gpio);

  memset(&huart1_link, 0, sizeof(huart1_link));
  huart1_link.Instance = USART1;
  huart1_link.Init.BaudRate = LINK_BAUD;
  huart1_link.Init.WordLength = UART_WORDLENGTH_8B;
  huart1_link.Init.StopBits = UART_STOPBITS_1;
  huart1_link.Init.Parity = UART_PARITY_NONE;
  huart1_link.Init.Mode = UART_MODE_TX_RX;
  huart1_link.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1_link.Init.OverSampling = UART_OVERSAMPLING_16;

  if (HAL_UART_Init(&huart1_link) != HAL_OK) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR USART1 init"));
#endif
    while (true) delay(1000);
  }
}

// --------------------------------------------------
// USART1 RX DMA init
// --------------------------------------------------
static void link_hal_dma_rx_init() {
  __HAL_RCC_DMA2_CLK_ENABLE();

  memset(&hdma_usart1_rx, 0, sizeof(hdma_usart1_rx));
  hdma_usart1_rx.Instance = DMA2_Stream2;
  hdma_usart1_rx.Init.Channel = DMA_CHANNEL_4;
  hdma_usart1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
  hdma_usart1_rx.Init.PeriphInc = DMA_PINC_DISABLE;
  hdma_usart1_rx.Init.MemInc = DMA_MINC_ENABLE;
  hdma_usart1_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
  hdma_usart1_rx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
  hdma_usart1_rx.Init.Mode = DMA_NORMAL;
  hdma_usart1_rx.Init.Priority = DMA_PRIORITY_HIGH;
  hdma_usart1_rx.Init.FIFOMode = DMA_FIFOMODE_DISABLE;

  HAL_DMA_DeInit(&hdma_usart1_rx);
  if (HAL_DMA_Init(&hdma_usart1_rx) != HAL_OK) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR DMA init"));
#endif
    while (true) delay(1000);
  }

  __HAL_LINKDMA(&huart1_link, hdmarx, hdma_usart1_rx);
}

// --------------------------------------------------
// RX DMA control
// --------------------------------------------------
static void clear_link_uart_errors() {
  huart1_link.ErrorCode = HAL_UART_ERROR_NONE;
#if defined(__HAL_UART_CLEAR_OREFLAG)
  __HAL_UART_CLEAR_OREFLAG(&huart1_link);
#endif
#if defined(__HAL_UART_CLEAR_NEFLAG)
  __HAL_UART_CLEAR_NEFLAG(&huart1_link);
#endif
#if defined(__HAL_UART_CLEAR_FEFLAG)
  __HAL_UART_CLEAR_FEFLAG(&huart1_link);
#endif
#if defined(__HAL_UART_CLEAR_PEFLAG)
  __HAL_UART_CLEAR_PEFLAG(&huart1_link);
#endif
}

static bool start_link_rx_dma(uint8_t *dst, uint16_t n) {
  HAL_UART_DMAStop(&huart1_link);
  clear_link_uart_errors();

  HAL_StatusTypeDef st = HAL_UART_Receive_DMA(&huart1_link, dst, n);

#if ENABLE_USB_DEBUG
  if (st != HAL_OK) {
    Serial.print(F("ERR HAL_UART_Receive_DMA st="));
    Serial.println((int)st);
  }
#endif

  return st == HAL_OK;
}

static bool wait_link_rx_dma(uint32_t timeout_ms) {
  uint32_t t0 = millis();

  delay(1);

  while (true) {
    if (huart1_link.ErrorCode != HAL_UART_ERROR_NONE) {
#if ENABLE_USB_DEBUG
      Serial.print(F("ERR UART ErrorCode="));
      Serial.println((uint32_t)huart1_link.ErrorCode);
#endif
      HAL_UART_DMAStop(&huart1_link);
      return false;
    }

    if (__HAL_DMA_GET_COUNTER(&hdma_usart1_rx) == 0) {
      HAL_UART_DMAStop(&huart1_link);
      return true;
    }

    if ((millis() - t0) > timeout_ms) {
#if ENABLE_USB_DEBUG
      Serial.println(F("ERR DMA RX timeout"));
#endif
      HAL_UART_DMAStop(&huart1_link);
      return false;
    }

    delay(1);
  }
}

// --------------------------------------------------
// Allocation and common setup
// --------------------------------------------------
static void init_noodle() {
  CONV5 = peaknet_l05_make();
  CONV6 = peaknet_l06_make();
  noodle_tensor_init(&TENSOR_IN);
  noodle_tensor_init(&TENSOR_TMP);

  if (!noodle_tensor_require_1d(&TENSOR_IN, IN_CH, L) ||
      !noodle_tensor_require_1d(&TENSOR_TMP, MID_CH, L)) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERROR: Noodle tensor allocation failed"));
#endif
    while (true) delay(1000);
  }
  noodle_tensor_set_quantization(&TENSOR_IN, CONV5.input_scale, CONV5.input_zero_point);
}

static void setup_common() {
  activity_led_init();
  ready_gpio_init();

#if ENABLE_USB_DEBUG
  Serial.begin(115200);
  delay(500);
  Serial.print(F("Distributed PeakNet1D Node "));
  Serial.println(NODE_ID);
#endif

#if USE_INTERMCU_ACK
  Ack.begin(ACK_BAUD);
#endif

  link_hal_uart1_init();
  link_hal_dma_rx_init();
  init_noodle();
}

// --------------------------------------------------
// Layer weights
// --------------------------------------------------


// --------------------------------------------------
// Recovery helpers
// --------------------------------------------------
static uint8_t recovery_events = 0;

static void node5_recover_or_reset(const __FlashStringHelper *reason) {
#if ENABLE_USB_DEBUG
  Serial.print(F("Node5 recovery: "));
  Serial.println(reason);
#endif

  activity_led_off();

  recovery_events++;
  if (recovery_events >= NODE5_MAX_RECOVERY_EVENTS) {
#if ENABLE_USB_DEBUG
    Serial.println(F("Node5 recovery limit reached; resetting"));
    Serial.flush();
#endif
    delay(50);
    NVIC_SystemReset();
  }
}

static bool node5_arm_rx_and_advertise_ready() {
  if (!start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    node5_recover_or_reset(F("failed to arm RX DMA"));
    return false;
  }

  // READY means Node5 RX DMA is already armed.
  ack_send_ready_upstream();
  return true;
}

static void node5_mark_healthy() {
  recovery_events = 0;
}

// --------------------------------------------------
// Arduino setup / loop
// --------------------------------------------------
void setup() {
  setup_common();

  // Node5 starts empty. Arm DMA for the first Conv4 tensor, then tell Node4
  // it is safe to send.
  if (start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    ack_send_ready_upstream();
  }
}

void loop() {
  // --------------------------------------------------
  // 1) Wait for current tensor from Node4.
  // --------------------------------------------------
  if (!wait_link_rx_dma(NODE5_RX_TIMEOUT_MS)) {
    node5_recover_or_reset(F("RX timeout"));
    node5_arm_rx_and_advertise_ready();
    return;
  }

  // Current input buffer is now occupied; deassert READY until RX is armed again.
  ready_set_upstream(false);

  uint32_t frame_id = 0;
  if (!decode_tensor_raw(RX_RAW_NEXT, &TENSOR_IN, L, IN_CH, CONV5.input_scale, CONV5.input_zero_point, &frame_id)) {
    node5_recover_or_reset(F("bad tensor packet"));
    node5_arm_rx_and_advertise_ready();
    return;
  }

  activity_led_on();

  uint16_t V = noodle_conv1d(&TENSOR_IN, &TENSOR_TMP, CONV5);
  if (V != L) {
    node5_recover_or_reset(F("INT8 Conv5 failed"));
    node5_arm_rx_and_advertise_ready();
    activity_led_off();
    return;
  }

  V = noodle_conv1d(&TENSOR_TMP, &TENSOR_IN, CONV6);
  if (V != L) {
    node5_recover_or_reset(F("INT8 Conv6 failed"));
    node5_arm_rx_and_advertise_ready();
    activity_led_off();
    return;
  }

  if (noodle_sigmoid(&TENSOR_IN) != L) {
    node5_recover_or_reset(F("INT8 sigmoid failed"));
    node5_arm_rx_and_advertise_ready();
    activity_led_off();
    return;
  }

  // --------------------------------------------------
  // 2) Arm next Node4 input before telling Node4 to send again.
  // --------------------------------------------------
  if (!start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    node5_recover_or_reset(F("failed to arm next RX"));
    return;
  }

  ack_send_ready_upstream();

  // --------------------------------------------------
  // 3) Send current scores to PC.
  // --------------------------------------------------
  if (!send_scores_binary_hal(&TENSOR_IN, V, frame_id)) {
    node5_recover_or_reset(F("send scores failed"));
    return;
  }

  // --------------------------------------------------
  // 4) Host ACK is useful, but should not stall forever.
  //    If it is missed, drop the ACK wait and continue.
  // --------------------------------------------------
  if (!wait_host_score_ack(NODE5_HOST_ACK_TIMEOUT_MS)) {
    node5_recover_or_reset(F("host score ACK timeout; continuing"));
    return;
  }

  activity_led_off();
  node5_mark_healthy();
}

