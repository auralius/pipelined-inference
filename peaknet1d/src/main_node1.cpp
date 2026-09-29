/**
 * Node1 HAL USART1 RX-DMA prefetch -- new Noodle full-INT8 Conv1.
 * This version uses USART1 RX with DMA to prefetch the next ECG packet while processing the current one. 
 * 
 * Forward data path:
 *   USART1 RX = PA10, USART1 TX = PA9, pure HAL
 * ACK path:
 *   Ack RX = PA3, Ack TX = PA2, HardwareSerial
 *
 * Behavior:
 *   setup(): arm RX DMA for first ECG packet. Do NOT send initial R because
 *            current Python already sends frame 1 using an initial permit.
 *   loop():
 *      wait DMA frame
 *      decode Float32 ECG and quantize into an INT8 NoodleTensor
 *      compute Conv1 using the new NoodleTensor API
 *      arm DMA for next ECG
 *      send R upstream
 *      wait Node2 R
 *      send current TNI (INT8 tensor) to Node2 using HAL blocking TX
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

#include "conv01_int8.h"

static const uint16_t L = 256;

#ifndef LINK_BAUD
#define LINK_BAUD 576000
#endif

#ifndef NODE_ID
#define NODE_ID 1
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
// GPIO READY input from Node2
// --------------------------------------------------
// Node2 PB0 -> Node1 PB1
// All GND common.
#ifndef USE_GPIO_READY
#define USE_GPIO_READY 1
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
// RX timeout: waiting for one complete host ECG packet.
// Downstream READY timeout: waiting for Node2 to accept the current tensor.
// If recovery repeats too many times, reset Node1 to avoid permanent stall.
#ifndef NODE1_RX_TIMEOUT_MS
#define NODE1_RX_TIMEOUT_MS 10000
#endif

#ifndef NODE1_DOWNSTREAM_TIMEOUT_MS
#define NODE1_DOWNSTREAM_TIMEOUT_MS 3000
#endif

#ifndef NODE1_MAX_RECOVERY_EVENTS
#define NODE1_MAX_RECOVERY_EVENTS 8
#endif

static NoodleTensor TENSOR_IN;   // quantized ECG, [1][256]
static NoodleTensor TENSOR_OUT;  // Conv1 output, [8][256]
static ConvMem CONV1;

// Host protocol is intentionally unchanged for apples-to-apples testing:
// "ECG" + uint32 frame_id + 256 float32 samples.
static uint8_t RX_RAW_NEXT[3 + 4 + L * 4];

static UART_HandleTypeDef huart1_link;
static DMA_HandleTypeDef hdma_usart1_rx;

#if USE_INTERMCU_ACK
HardwareSerial Ack(PA3, PA2);
#endif

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
// GPIO downstream READY helper
// --------------------------------------------------
static void downstream_ready_gpio_init() {
#if USE_GPIO_READY
  pinMode(READY_IN_DOWNSTREAM_PIN, INPUT_PULLDOWN);
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
  // Node1 upstream READY still goes to the PC/host through the existing ACK UART.
  Ack.write((uint8_t)'R');
  Ack.flush();
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

static uint32_t read_u32_le(const uint8_t *p) {
  return ((uint32_t)p[0]) |
         ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static float read_f32_le(const uint8_t *p) {
  float f;
  memcpy(&f, p, 4);
  return f;
}

static bool decode_ecg_raw(const uint8_t *raw, NoodleTensor *tensor, uint32_t *frame_id) {
  if (raw[0] != 'E' || raw[1] != 'C' || raw[2] != 'G') {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERR bad ECG header"));
#endif
    return false;
  }

  if (frame_id) *frame_id = read_u32_le(raw + 3);

  NoodleData *x = noodle_tensor_require_1d(tensor, 1, L);
  if (!x) return false;
  noodle_tensor_set_quantization(tensor, CONV1.input_scale, CONV1.input_zero_point);

  const uint8_t *p = raw + 7;
  for (uint16_t i = 0; i < L; ++i, p += 4) {
    const float f = read_f32_le(p);
    x[i] = (NoodleData)noodle_quantize_float(
        f, CONV1.input_scale, CONV1.input_zero_point);
  }
  return true;
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

static bool send_tensor_hal(const NoodleTensor *tensor,
                            uint16_t length, uint16_t channels,
                            uint32_t frame_id) {
  if (!tensor || !noodle_tensor_const_data(tensor)) return false;
  if (tensor->rank != NOODLE_TENSOR_1D || tensor->W != length || tensor->C != channels) return false;

  // TNI = Tensor, Noodle INT8. Deliberately distinct from legacy Float32 TNS.
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

  // Give DMA at least one scheduler tick before checking counter.
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
  CONV1 = peaknet_l01_make();
  noodle_tensor_init(&TENSOR_IN);
  noodle_tensor_init(&TENSOR_OUT);

  if (!noodle_tensor_require_1d(&TENSOR_IN, 1, L) ||
      !noodle_tensor_require_1d(&TENSOR_OUT, 8, L)) {
#if ENABLE_USB_DEBUG
    Serial.println(F("ERROR: Noodle tensor allocation failed"));
#endif
    while (true) delay(1000);
  }
  noodle_tensor_set_quantization(&TENSOR_IN, CONV1.input_scale, CONV1.input_zero_point);
}

static void setup_common() {
  activity_led_init();
  downstream_ready_gpio_init();

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
// Recovery helpers
// --------------------------------------------------
static uint8_t recovery_events = 0;

static void node1_recover_or_reset(const __FlashStringHelper *reason) {
#if ENABLE_USB_DEBUG
  Serial.print(F("Node1 recovery: "));
  Serial.println(reason);
#endif

  activity_led_off();

  recovery_events++;
  if (recovery_events >= NODE1_MAX_RECOVERY_EVENTS) {
#if ENABLE_USB_DEBUG
    Serial.println(F("Node1 recovery limit reached; resetting"));
    Serial.flush();
#endif
    delay(50);
    NVIC_SystemReset();
  }
}

static bool node1_arm_rx_and_advertise_ready() {
  if (!start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) { // if dma keeps failing, then reboot
    node1_recover_or_reset(F("failed to arm RX DMA"));
    return false;
  }

  // Once RX DMA is armed, it is safe to allow the host to send.
  ack_send_ready_upstream();
  return true;
}

static void node1_mark_healthy() {
  recovery_events = 0;
}

void setup() {
  setup_common();

  // Existing Python sends frame 1 using an initial implicit permit.
  // Therefore, arm DMA but do not send initial READY here.
  start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT));
}

void loop() {
  // --------------------------------------------------
  // 1) Wait for current ECG packet from the host.
  // --------------------------------------------------
  if (!wait_link_rx_dma(NODE1_RX_TIMEOUT_MS)) {
    // Node1 is empty again after timeout. Re-arm RX and advertise READY
    // so the host can send/retry the next frame instead of waiting forever.
    node1_recover_or_reset(F("RX timeout"));
    node1_arm_rx_and_advertise_ready();
    return;
  }

  uint32_t frame_id = 0;
  if (!decode_ecg_raw(RX_RAW_NEXT, &TENSOR_IN, &frame_id)) {
    // Bad header means the host/Node1 packet alignment is wrong.
    // Drop this packet, re-arm RX, and advertise READY again.
    node1_recover_or_reset(F("bad ECG header"));
    node1_arm_rx_and_advertise_ready();
    return;
  }

  activity_led_on();

  const unsigned long st = micros();

  uint16_t V = noodle_conv1d(&TENSOR_IN, &TENSOR_OUT, CONV1);
  if (V != L) {
    node1_recover_or_reset(F("INT8 Conv1 failed"));
    node1_arm_rx_and_advertise_ready();
    activity_led_off();
    return;
  }

#if ENABLE_USB_DEBUG
  float et = (float)(micros() - st) * 1e-6f;
  Serial.print(F("Node1 Conv1 sec="));
  Serial.println(et, 6);
#else
  (void)st;
#endif

  // --------------------------------------------------
  // 2) Prefetch the next host ECG while this frame waits downstream.
  //    RX_RAW_NEXT is independent from the NoodleTensor activations, so it is
  //    safe to arm DMA now. READY is asserted only after DMA is armed.
  // --------------------------------------------------
  if (!start_link_rx_dma(RX_RAW_NEXT, sizeof(RX_RAW_NEXT))) {
    node1_recover_or_reset(F("failed to arm next RX"));
    activity_led_off();
    return;
  }
  ack_send_ready_upstream();

  // --------------------------------------------------
  // 3) Wait for Node2. If Node2 does not become ready, drop this output.
  //    The next host RX is already armed, so the input side remains alive.
  // --------------------------------------------------
  if (!ack_wait_ready_downstream(NODE1_DOWNSTREAM_TIMEOUT_MS)) {
    node1_recover_or_reset(F("Node2 READY timeout; dropping current tensor"));
    activity_led_off();
    return;
  }

  // --------------------------------------------------
  // 4) Send current Conv1 tensor downstream.
  // --------------------------------------------------
  if (!send_tensor_hal(&TENSOR_OUT, V, 8, frame_id)) {
    node1_recover_or_reset(F("send tensor failed"));
    activity_led_off();
    return;
  }

  // DONE!
  activity_led_off();
  node1_mark_healthy();
}

