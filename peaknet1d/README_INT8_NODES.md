# PeakNet distributed nodes — new Noodle full-INT8 build

This package rewrites Nodes 1–5 around the current `NoodleTensor` API and the
full-INT8 parameter export generated from `ecg_peak_detect_new_noodle_int8.ipynb`.

## Important: these are project-wide build flags

The macros must be visible while compiling **every Noodle `.cpp` file**, not only
`main_nodeX.cpp`:

```ini
build_flags =
    -D NOODLE_USE_INT8
    -D NOODLE_MAX_K=9
    -D NOODLE_BUFFER_ARENA_INITIAL_BYTES=8192
    -D NOODLE_USE_NONE
```

`NOODLE_MAX_K=9` is required by PeakNet Conv1.  The 8192-byte initial arena is
large enough for the largest pair of INT8 activation tensors used by Nodes 3–5
and avoids repeated arena growth during startup. `NOODLE_USE_NONE` disables the
filesystem backend because these nodes use memory-backed generated parameters;
it also avoids an unnecessary SdFat dependency.

## What stays compatible

The host-side protocol is intentionally unchanged:

- Host -> Node1: `"ECG" + uint32 frame_id + 256 float32 samples` (1031 bytes)
- Node5 -> host: `"SCB" + uint32 frame_id + uint16 length + 256 float32 scores`

Node1 quantizes the incoming ECG using Conv1's calibrated input scale/zero point.
Node5 runs the INT8 sigmoid, then dequantizes probabilities before transmitting
legacy Float32 SCB scores.  The existing Python benchmark therefore does not
need to change for the first INT8 comparison.

## New MCU-to-MCU packet

The four internal links use a deliberately different magic value so a Float32
node cannot silently interpret an INT8 payload:

```text
"TNI" + uint32 frame_id + uint16 length + uint16 channels + int8 payload
```

Packet sizes:

| Link | Shape | Legacy TNS Float32 | New TNI INT8 |
|---|---:|---:|---:|
| Node1 -> Node2 | 256 x 8  | 8203 bytes | 2059 bytes |
| Node2 -> Node3 | 256 x 16 | 16395 bytes | 4107 bytes |
| Node3 -> Node4 | 256 x 16 | 16395 bytes | 4107 bytes |
| Node4 -> Node5 | 256 x 16 | 16395 bytes | 4107 bytes |

(The legacy sizes include the 11-byte frame-ID/shape header.)

No scale or zero point is sent on the wire.  The quantization domains are fixed
by the trained model, and each generated layer header contains the expected
input/output quantization parameters.  Adjacent layer domains from the exported
model match exactly.

## Node mapping

- Node 1: `conv01_int8.h`
- Node 2: `conv02_int8.h`
- Node 3: `conv03_int8.h`
- Node 4: `conv04_int8.h`
- Node 5: `conv05_int8.h` + `conv06_int8.h` + `noodle_sigmoid()`

## Files

- `src/main_node1.cpp` ... `src/main_node5.cpp`: rewritten firmware
- `include/conv0X_int8.h`: generated parameters and requantization metadata
- `lib/Noodle/src/`: exact new Noodle revision supplied for this migration

The five `main_node*.cpp` files are alternatives, just like the original source
set.  Build/flash one node source at a time; do not compile all five into one
firmware image.
