/**
 * Node4 HAL USART1 RX-DMA prefetch -- new Noodle full-INT8 Conv4.
 *
 * Forward data path:
 *   USART1 RX = PA10, USART1 TX = PA9, pure HAL
 * ACK path:
 *   Ack RX = PA3, Ack TX = PA2, HardwareSerial
 *
 * Behavior:
 *   setup():
 *      arm RX DMA for first tensor from Node3
 *      send initial READY upstream to Node3
 *
 *   loop():
 *      wait DMA tensor
 *      decode TNI directly into an INT8 NoodleTensor
 *      compute Conv4 using the new NoodleTensor API
 *      arm DMA for next tensor
 *      send READY upstream to Node3
 *      wait Node5 READY
 *      send current tensor to Node5 using HAL blocking TX
 *
 * Important rule:
 *   READY is sent only after the next RX DMA buffer is armed.
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

#include "conv04_int8.h"

static const uint16_t L = 256;
static const uint16_t IN_CH = 16;
static const uint16_t OUT_CH = 16;

#ifndef LINK_BAUD
#define LINK_BAUD 576000
#endif

#ifndef NODE_ID
#define NODE_ID 4
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
// GPIO READY/ACK flow-control pins
// --------------------------------------------------
// READY is active-high by default.
// Wire downstream READY_OUT_UPSTREAM_PIN to upstream READY_IN_DOWNSTREAM_PIN.
//
// Recommended wiring for the 5-node chain:
//   Node2 PB0 -> Node1 PB1
//   Node3 PB0 -> Node2 PB1
//   Node4 PB0 -> Node3 PB1
//   Node5 PB0 -> Node4 PB1
//   All GND must be common.
#ifndef USE_GPIO_READY
#define USE_GPIO_READY 1
#endif

#ifndef READY_OUT_UPSTREAM_PIN
#define READY_OUT_UPSTREAM_PIN PB0
#endif

#ifndef READY_IN_DOWNSTREAM_PIN
#define READY_IN_DOWNSTREAM_PIN PB1
#endif

#ifndef READY_ACTIVE_LEVEL
#define READY_ACTIVE_LEVEL HIGH
#endif


// --------------------------------------------------
// Recovery configuration
// --------------------------------------------------
#ifndef NODE4_RX_TIMEOUT_MS
#define NODE4_RX_TIMEOUT_MS 10000
#endif

#ifndef NODE4_DOWNSTREAM_TIMEOUT_MS
#define NODE4_DOWNSTREAM_TIMEOUT_MS 3000
#endif

#ifndef NODE4_MAX_RECOVERY_EVENTS
#define NODE4_MAX_RECOVERY_EVENTS 8
#endif

static NoodleTensor TENSOR_IN;
static NoodleTensor TENSOR_OUT;
static ConvMem CONV4;

// TNI packet: "TNI" + frame_id + shape + signed INT8 payload.
static uint8_t RX_RAW_NEXT[3 + 4 + 2 + 2 + L * IN_CH];

static UART_HandleTypeDef huart1_link;
static DMA_HandleTypeDef hdma_usart1_rx;

// Inter-MCU READY now uses GPIO PB0/PB1, so PA2/PA3 are no longer used for node-to-node ACK.

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
// GPIO READY helpers
// --------------------------------------------------
static void ready_gpio_init() {
#if USE_GPIO_READY
  pinMode(READY_OUT_UPSTREAM_PIN, OUTPUT);
  digitalWrite(READY_OUT_UPSTREAM_PIN, (READY_ACTIVE_LEVEL == HIGH) ? LOW : HIGH);

  pinMode(READY_IN_DOWNSTREAM_PIN, INPUT_PULLDOWN);
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

static bool downstream_ready_is_high() {
#if USE_GPIO_READY
  return digitalRead(READY_IN_DOWNSTREAM_PIN) == READY_ACTIVE_LEVEL;
#else
  return true;
#endif
}

static void ack_send_ready_upstream() {
#if USE_INTERMCU_ACK
  // READY means this node has already armed RX DMA and can accept the next tensor.
  ready_set_upstream(true);
#endif
}

static bool ack_wait_ready_downstream(uint32_t timeout_ms) {
#if USE_INTERMCU_ACK
  uint32_t t0 = millis();
  while ((millis() - t0) <= timeout_ms) {
    if (downstream_ready_is_high()) return true;
    delay(1);
  }
#if ENABLE_USB_DEBUG
  Serial.println(F("ERR downstream GPIO READY timeout"));
#endif
  return false;
#else
  (void)timeout_ms;
  return true;
#endif
}

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
  uint8_t b[2] = {(uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF)};
  return link_tx_bytes(b, 2);
}

static bool link_tx_u32_le(uint32_t v) {
  uint8_t b[4] = {
    (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF),
    (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF)
  };
  return link_tx_bytes(b, 4);
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

static bool send_tensor_hal(const NoodleTensor *tensor,
                            uint16_t length, uint16_t channels,
                            uint32_t frame_id) {
  if (!tensor || !noodle_tensor_const_data(tensor)) return false;
  if (tensor->rank != NOODLE_TENSOR_1D || tensor->W != length || tensor->C != channels) return false;

  const uint8_t hdr[3] = {'T', 'N', 'I'};
  if (!link_tx_bytes(hdr, 3)) return false;
  if (!link_tx_u32_le(frame_id)) return false;
  if (!link_tx_u16_le(length)) return false;
  if (!link_tx_u16_le(channels)) return false;

  const uint32_t n = (uint32_t)length * (uint32_t)channels;
  return link_tx_bytes((const uint8_t*)noodle_tensor_const_data(tensor), (uint16_t)n, 2000);
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

static void init_noodle() {
  CONV4 = peaknet_l04_make();
  noodle_tensor_init(&TENSOR_IN);
  noodle_tensor_init(&TENSOR_OUT);

  if (!noodle_tensor_require_1d(&TENSOR_IN, IN_CH, L) ||
      !noodle_tensor_require_1d(&TENSOR_OUT, OUT_CH, L)) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERROR: Noodle tensor allocation failed"));
#endif
    while (true) delay(1000);
  }
  noodle_tensor_set_quantization(&TENSOR_IN, CONV4.input_scale, CONV4.input_zero_point);
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


  link_hal_uart1_init();
  link_hal_dma_rx_init();
  init_noodle();
}



// --------------------------------------------------
// Recovery helpers
// --------------------------------------------------
static uint8_t recovery_events = 0;

static void node4_recover_or_reset(const __FlashStringHelper *reason) {
#if ENABLE_USB_DEBUG
  Serial.print(F("Node4 recovery: "));
  Serial.println(reason);
#endif

  activity_led_off();

  recovery_events++;
  if (recovery_events >= NODE4_MAX_RECOVERY_EVENTS) {
#if ENABLE_USB_DEBUG
    Serial.println(F("Node4 recovery limit reached; resetting"));
    Serial.flush();
#endif
    delay(50);
    NVIC_SystemReset();
  }
}

static bool node4_arm_rx_and_advertise_ready() {
  if (!start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    node4_recover_or_reset(F("failed to arm RX DMA"));
    return false;
  }

  // READY means this node's RX DMA is already armed.
  ack_send_ready_upstream();
  return true;
}

static void node4_mark_healthy() {
  recovery_events = 0;
}

void setup() {
  setup_common();

  // Node4 starts empty. Arm DMA for the first tensor, then tell Node3
  // it is safe to send.
  if (start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    ack_send_ready_upstream();
  }
}

void loop() {
  // --------------------------------------------------
  // 1) Wait for current tensor from Node3.
  // --------------------------------------------------
  if (!wait_link_rx_dma(NODE4_RX_TIMEOUT_MS)) {
    // Receiver is empty again after timeout. Re-arm RX and advertise READY
    // upstream so Node3 does not wait forever.
    node4_recover_or_reset(F("RX timeout"));
    node4_arm_rx_and_advertise_ready();
    return;
  }

  // Current input buffer is now occupied; deassert READY until RX is armed again.
  ready_set_upstream(false);

  uint32_t frame_id = 0;
  if (!decode_tensor_raw(RX_RAW_NEXT, &TENSOR_IN, L, IN_CH, CONV4.input_scale, CONV4.input_zero_point, &frame_id)) {
    // Bad packet/header means alignment or corruption issue.
    // Drop this packet and re-arm RX.
    node4_recover_or_reset(F("bad tensor packet"));
    node4_arm_rx_and_advertise_ready();
    return;
  }

  activity_led_on();

  const unsigned long st = micros();

  uint16_t V = noodle_conv1d(&TENSOR_IN, &TENSOR_OUT, CONV4);
  if (V != L) {
    node4_recover_or_reset(F("INT8 Conv4 failed"));
    node4_arm_rx_and_advertise_ready();
    activity_led_off();
    return;
  }

#if ENABLE_USB_DEBUG
  float et = (float)(micros() - st) * 1e-6f;
  Serial.print(F("Node4 Conv4 sec="));
  Serial.println(et, 6);
#else
  (void)st;
#endif

  // --------------------------------------------------
  // 2) Arm next input before telling Node3 to send again.
  // --------------------------------------------------
  if (!start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    node4_recover_or_reset(F("failed to arm next RX"));
    return;
  }

  ack_send_ready_upstream();

  // --------------------------------------------------
  // 3) Wait for Node5. If it does not become ready, drop the
  //    current output tensor and keep the upstream side alive.
  // --------------------------------------------------
  if (!ack_wait_ready_downstream(NODE4_DOWNSTREAM_TIMEOUT_MS)) {
    node4_recover_or_reset(F("downstream READY timeout; dropping current tensor"));
    return;
  }

  // --------------------------------------------------
  // 4) Send current tensor downstream.
  // --------------------------------------------------
  if (!send_tensor_hal(&TENSOR_OUT, V, OUT_CH, frame_id)) {
    node4_recover_or_reset(F("send tensor failed"));
    return;
  }

  activity_led_off();
  node4_mark_healthy();
}

