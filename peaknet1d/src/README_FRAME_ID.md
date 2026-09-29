# Frame-ID protocol version for 5-MCU PeakNet chain

This version adds a 32-bit little-endian `frame_id` to every packet so the host no longer has to assume FIFO mapping.

## Packet formats

### Host -> Node1 ECG packet

```text
"ECG" + uint32 frame_id + 256 float32 samples
```

Packet length: `3 + 4 + 256*4 = 1031 bytes`.

### Node-to-node TNS packet

```text
"TNS" + uint32 frame_id + uint16 length + uint16 channels + float32 payload
```

For `256 x 16`, packet length: `3 + 4 + 2 + 2 + 256*16*4 = 16395 bytes`.

### Node5 -> Host score packet

```text
"SCB" + uint32 frame_id + uint16 length + 256 float32 scores
```

Packet length: `3 + 4 + 2 + 256*4 = 1033 bytes`.

## What changed

- Node1 decodes `frame_id` from ECG packets and forwards it in TNS.
- Nodes 2, 3, and 4 decode and forward the same `frame_id`.
- Node5 decodes the incoming `frame_id` and includes it in the SCB score packet.
- Python sends `frame_id = 1, 2, 3, ...` and maps received scores by actual `frame_id`, not by FIFO order.
- If a score arrives for an unknown `frame_id`, Python prints a warning and shows pending frame IDs.

## Test recommendation

Start with the stable setting:

```bash
python3 ecg-sender-frame-id.py --tx-port /dev/ttyUSB0 --rx-port /dev/ttyUSB1 --baud 576000 --mode fast --duration 30 --max-pending 4
```

Then test the problematic case:

```bash
python3 ecg-sender-frame-id.py --tx-port /dev/ttyUSB0 --rx-port /dev/ttyUSB1 --baud 576000 --mode fast --duration 30 --max-pending 6
```

With frame IDs, you can distinguish:

- missing frame IDs: frame lost or stuck in pipeline,
- duplicate frame IDs: stale output resent,
- unknown frame IDs: corruption or parser desynchronization,
- correct frame IDs but shifted peaks: model/window alignment issue rather than packet mapping.
