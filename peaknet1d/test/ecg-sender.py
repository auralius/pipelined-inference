#!/usr/bin/env python3
"""
PeakNet 5-MCU sender with Node1 host READY, non-blocking RX/READY polling.

Why this version:
- The previous version blocked while waiting for Node1 'R'.
- During that blocking wait, Python did not continuously drain Node5 SCORES.
- Long SCORES lines can then accumulate in the serial/OS buffer and cause intermittent behavior.
- This version polls both ports continuously:
    * FTDI #1 RXD receives Node1 READY byte 'R'
    * FTDI #2 RXD receives Node5 SCORES lines

Wiring:
- FTDI #1 TXD -> Node1 PA10  : Python sends ECG frames
- Node1 PA2   -> FTDI #1 RXD : Node1 sends one byte 'R' when ready
- Node1 PA9 -> Node2 PA10 -> ... -> Node5
- Node5 PA9 -> FTDI #2 RXD  : Python receives SCORES
- FTDI #2 TXD -> Node5 PA3  : Python sends score ACK byte b'A'

Node1 protocol:
- Python sends frame 1 immediately.
- Node1 sends b'R' after it has forwarded the previous Conv1 tensor to Node2.
- Python sends frame k+1 only after receiving one 'R' permit.

Node5 protocol:
- RX binary packet: b"SCB" + uint16 length + 256 float32 scores
- After each valid packet, Python writes b"A" to FTDI #2 TXD as result ACK

Example:
python3 ecg-sender-binary-scores-maxpending.py \
  --tx-port /dev/ttyUSB0 \
  --rx-port /dev/ttyUSB1 \
  --baud 576000 \
  --mode fast \
  --duration 30 \
  --max-pending 6
"""

import argparse
import time
import struct
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt


DEFAULT_TX_PORT = "/dev/ttyUSB0"
DEFAULT_RX_PORT = "/dev/ttyUSB1"
DEFAULT_BAUD = 576000

FS = 256
L = 256
HOP = 256

THRESH = 0.5
REFRACTORY_S = 0.25


def synthetic_ecg(fs=256, duration_s=20.0, hr_bpm=72, noise=0.02, seed=2):
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs

    rr = 60.0 / hr_bpm
    beats = np.arange(0.5, duration_s - 0.5, rr)

    x = np.zeros_like(t)

    def add_gaussian(center, amp, width):
        nonlocal x
        x += amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    x += 0.05 * np.sin(2 * np.pi * 0.33 * t)

    for bt in beats:
        add_gaussian(bt - 0.18,  0.12, 0.035)
        add_gaussian(bt - 0.04, -0.15, 0.010)
        add_gaussian(bt,        1.00, 0.012)
        add_gaussian(bt + 0.03, -0.25, 0.012)
        add_gaussian(bt + 0.22,  0.35, 0.060)

    x += noise * rng.standard_normal(size=n)
    x /= (np.max(np.abs(x)) + 1e-9)
    return t, x, beats


def score_to_peaks(score, fs, thresh=0.5, refractory_s=0.25):
    score = np.asarray(score)
    refractory = max(1, int(refractory_s * fs))

    idx = np.where(score > thresh)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)

    peaks = []
    start = idx[0]
    prev = idx[0]

    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue

        region = np.arange(start, prev + 1)
        peaks.append(region[np.argmax(score[region])])
        start = k
        prev = k

    region = np.arange(start, prev + 1)
    peaks.append(region[np.argmax(score[region])])

    peaks = np.array(peaks, dtype=int)
    kept = [peaks[0]]

    for p in peaks[1:]:
        if p - kept[-1] >= refractory:
            kept.append(p)
        elif score[p] > score[kept[-1]]:
            kept[-1] = p

    return np.array(kept, dtype=int)


def normalize_frame(frame):
    frame = np.asarray(frame, dtype=np.float32)
    return (frame - frame.mean()) / (frame.std() + 1e-6)


def send_frame(ser, frame_f32):
    frame_f32 = np.asarray(frame_f32, dtype=np.float32)
    if frame_f32.shape[0] != L:
        raise ValueError(f"frame length must be {L}, got {frame_f32.shape[0]}")
    ser.write(b"ECG" + frame_f32.tobytes(order="C"))
    ser.flush()


def drain_ready_bytes(ser):
    """
    Read all currently available bytes from Node1 READY port.
    Count 'R' bytes as permits. Ignore everything else.
    """
    n = ser.in_waiting
    if n <= 0:
        return 0

    data = ser.read(n)
    return data.count(b"R")



class BinaryScoreReader:
    """
    Nonblocking binary packet reader for Node5.

    Packet format:
      b"SCB" + uint16 length + float32 scores[length]
    For length=256, packet size = 3 + 2 + 1024 = 1029 bytes.
    """
    def __init__(self, ser, expected_length=256):
        self.ser = ser
        self.expected_length = expected_length
        self.buf = bytearray()

    def poll(self):
        n = self.ser.in_waiting
        if n > 0:
            self.buf.extend(self.ser.read(n))

        sync = b"SCB"
        idx = self.buf.find(sync)
        if idx < 0:
            if len(self.buf) > 2:
                del self.buf[:-2]
            return None

        if idx > 0:
            del self.buf[:idx]

        header_len = 3 + 2
        if len(self.buf) < header_len:
            return None

        length = struct.unpack_from("<H", self.buf, 3)[0]
        if length != self.expected_length:
            del self.buf[0]
            return None

        packet_len = header_len + 4 * length
        if len(self.buf) < packet_len:
            return None

        payload = self.buf[header_len:packet_len]
        scores = np.frombuffer(payload, dtype="<f4").copy()

        del self.buf[:packet_len]

        if not np.all(np.isfinite(scores)):
            print("[warn] binary SCORES contains NaN/Inf")

        return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-port", default=DEFAULT_TX_PORT,
                        help="FTDI #1: TXD to Node1 PA10 and RXD from Node1 PA2.")
    parser.add_argument("--rx-port", default=DEFAULT_RX_PORT,
                        help="FTDI #2: RXD from Node5 PA9.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--mode", choices=["fast", "realtime"], default="realtime")
    parser.add_argument("--plot-window", type=float, default=10.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESH)
    parser.add_argument("--permit-cap", type=int, default=1,
                        help="Cap accumulated Node1 READY permits. Keep 1 for single-buffer Node1.")
    parser.add_argument("--max-pending", type=int, default=6,
                        help="Maximum number of transmitted frames allowed to wait for SCORES.")
    parser.add_argument("--score-ack-byte", default="A",
                        help="One-byte ACK sent to Node5 after each binary SCORES packet.")
    args = parser.parse_args()

    if args.tx_port == args.rx_port:
        raise ValueError("tx-port and rx-port must be different.")

    score_ack = args.score_ack_byte.encode("latin1")
    if len(score_ack) != 1:
        raise ValueError("--score-ack-byte must encode to exactly one byte.")

    t, x, true_beats = synthetic_ecg(fs=FS, duration_s=args.duration)
    n = len(x)
    starts = list(range(0, n - L + 1, HOP))

    score_all = np.zeros(n, dtype=np.float32)
    count_all = np.zeros(n, dtype=np.float32)

    pending = deque()

    sent_count = 0
    recv_count = 0
    missed_count = 0
    next_tx_index = 0

    host_latency_times = []
    rx_arrival_times = []

    print(f"[host] opening Node1 TX/READY serial {args.tx_port} @ {args.baud} ...")
    tx_ser = serial.Serial(args.tx_port, args.baud, timeout=0.0, write_timeout=2.0)

    print(f"[host] opening Node5 RX serial {args.rx_port} @ {args.baud} ...")
    rx_ser = serial.Serial(args.rx_port, args.baud, timeout=0.001)

    time.sleep(1.5)
    tx_ser.reset_input_buffer()
    tx_ser.reset_output_buffer()
    rx_ser.reset_input_buffer()
    rx_ser.reset_output_buffer()

    score_reader = BinaryScoreReader(rx_ser, expected_length=L)

    host_t0 = time.perf_counter()

    # Initial permit: Node1 starts empty and is already waiting for the first ECG frame.
    ready_permits = 1

    print(f"[host] mode={args.mode}, frames={len(starts)}, max_pending={args.max_pending}")
    print("[host] first frame uses initial permit; later permits come from Node1 byte 'R'")
    print("[host] READY and binary SCORES are polled non-blockingly")
    print("[host] Python sends score ACK to Node5 after each received SCORES packet")

    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(t, x, label="ECG synthetic")
        score_line, = ax.plot(t, 0.6 * score_all - 0.8, label="score live")

        pred_scatter = ax.scatter([], [], marker="x", s=90, linewidths=2.5, label="Pred peaks")
        true_idx = (true_beats * FS).astype(int)
        true_idx = true_idx[(true_idx >= 0) & (true_idx < n)]
        ax.scatter(t[true_idx], x[true_idx], marker="o", s=60,
                   facecolors="none", edgecolors="black", label="True R ref")

        ax.set_title("Live PeakNet 5-MCU UART chain with Node1 READY")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
    else:
        fig = ax = score_line = pred_scatter = None

    last_plot_update = 0.0

    while next_tx_index < len(starts) or pending:
        now = time.perf_counter()

        # Always drain host READY bytes first, without blocking.
        new_permits = drain_ready_bytes(tx_ser)
        if new_permits:
            ready_permits = min(args.permit_cap, ready_permits + new_permits)
            print(f"[ready] +{new_permits}, permits={ready_permits}")

        # Always read Node5 binary scores without blocking.
        scores = score_reader.poll()
        line = None

        if scores is not None:
            # Acknowledge Node5 output after receiving one complete binary SCORES packet.
            rx_ser.write(score_ack)
            rx_ser.flush()

            recv_count += 1
            rx_arrival_times.append(time.perf_counter())

            if pending:
                item = pending.popleft()
                old_start = item["start"]
                tx_index = item["tx_index"]
                tx_t = item["tx_t"]

                rx_t = time.perf_counter()
                host_latency = rx_t - tx_t
                host_latency_times.append(host_latency)

                score_all[old_start:old_start + L] += scores
                count_all[old_start:old_start + L] += 1.0

                print(
                    f"[rx] {recv_count:04d} start={old_start} "
                    f"tx={tx_index:04d} "
                    f"argmax={int(np.argmax(scores))} "
                    f"max={float(np.nanmax(scores)):.3f} "
                    f"host_t={host_latency:.6f}s "
                    f"pending={len(pending)}"
                )
            else:
                missed_count += 1
                print("[warn] received SCORES but pending queue is empty")

        elif line:
            print("[dev-rx]", line)

        # TX path: send only if there is a permit and pending queue has room.
        can_send = (
            next_tx_index < len(starts)
            and ready_permits > 0
            and len(pending) < args.max_pending
        )

        if can_send and args.mode == "realtime":
            start = starts[next_tx_index]
            target_time = host_t0 + start / FS
            if now < target_time:
                can_send = False

        if can_send:
            start = starts[next_tx_index]
            frame = normalize_frame(x[start:start + L])

            tx_t = time.perf_counter()
            send_frame(tx_ser, frame)

            ready_permits -= 1
            next_tx_index += 1
            sent_count += 1

            pending.append({
                "start": start,
                "tx_index": sent_count,
                "tx_t": tx_t,
            })

            print(f"[tx] {sent_count:04d}/{len(starts)} start={start} pending={len(pending)} permits={ready_permits}")

        if not args.no_plot and (time.perf_counter() - last_plot_update) > 0.25:
            last_plot_update = time.perf_counter()

            count_safe = count_all.copy()
            count_safe[count_safe == 0] = 1.0
            score_avg = score_all / count_safe
            peaks = score_to_peaks(score_avg, FS, thresh=args.threshold, refractory_s=REFRACTORY_S)

            score_line.set_ydata(0.6 * score_avg - 0.8)

            if len(peaks) > 0:
                pred_scatter.set_offsets(np.column_stack([t[peaks], x[peaks]]))
            else:
                pred_scatter.set_offsets(np.empty((0, 2)))

            if args.mode == "realtime" and recv_count > 0:
                completed_idx = min(recv_count - 1, len(starts) - 1)
                latest_sample = min(n - 1, starts[completed_idx] + L)
                latest_time = t[latest_sample]
                left = max(0.0, latest_time - args.plot_window)
                right = max(args.plot_window, latest_time + 0.5)
                ax.set_xlim(left, right)
            else:
                ax.set_xlim(0, min(args.duration, max(10.0, t[-1])))

            ax.set_ylim(-0.9, 1.1)
            fig.canvas.draw()
            fig.canvas.flush_events()

        time.sleep(0.001)

    tx_ser.close()
    rx_ser.close()

    count_safe = count_all.copy()
    count_safe[count_safe == 0] = 1.0
    score_avg = score_all / count_safe

    peaks = score_to_peaks(score_avg, FS, thresh=args.threshold, refractory_s=REFRACTORY_S)
    true_idx = (true_beats * FS).astype(int)
    true_idx = true_idx[(true_idx >= 0) & (true_idx < n)]

    elapsed = time.perf_counter() - host_t0

    print("\n=== Live Sender Summary ===")
    print(f"Mode              : {args.mode}")
    print(f"Frames sent       : {sent_count}")
    print(f"Results received  : {recv_count}")
    print(f"Pending left      : {len(pending)}")
    print(f"Empty-queue RX    : {missed_count}")
    print(f"Wall time         : {elapsed:.3f} s")
    if elapsed > 0:
        print(f"Host result rate  : {recv_count / elapsed:.2f} results/s")


    if host_latency_times:
        print("Host TX-to-RX seconds (min/avg/max):",
              float(np.min(host_latency_times)),
              float(np.mean(host_latency_times)),
              float(np.max(host_latency_times)))

    if len(rx_arrival_times) >= 2:
        intervals = np.diff(np.array(rx_arrival_times))
        print("RX result interval seconds (min/avg/max):",
              float(np.min(intervals)),
              float(np.mean(intervals)),
              float(np.max(intervals)))
        print("RX steady result rate approx:",
              float(1.0 / np.mean(intervals)), "results/s")

    print("\n=== Peak Errors ===")
    m = min(len(peaks), len(true_idx))
    for i in range(m):
        err_samp = int(peaks[i] - true_idx[i])
        err_ms = 1000.0 * err_samp / FS
        print(f"{i:02d}: pred={peaks[i]:4d}, true={true_idx[i]:4d}, err={err_samp:+4d} samples ({err_ms:+6.1f} ms)")

    print(f"\nPred peaks: {len(peaks)}")
    print(f"True beats: {len(true_idx)}")

    if not args.no_plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
