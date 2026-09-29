# Pipelined Multi-Microcontroller Inference

## ECG Peak Detection

https://github.com/user-attachments/assets/27bac2d4-41a9-4625-b97a-aaa480c9cf8c

| Metric | 5-MCU | Single MCU |
|---|---:|---:|
| Inter-stage baud rate | 1 Mbit/s | – |
| Largest intermediate packet | 4,107 B | – |
| Frames sent | 30 | 30 |
| Results received | 28 | 30 |
| Observed missing results | 2 | 0 |
| Mean TX-to-RX latency | ~1.885 s | ~1.338 s |
| Steady-state interval | ~0.471 s | ~1.339 s |
| Steady-state throughput | ~2.121 frames/s | ~0.747 frames/s |
| Maximum in-flight depth | 6 | 1 |

