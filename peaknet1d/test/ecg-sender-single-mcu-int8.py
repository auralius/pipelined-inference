#!/usr/bin/env python3
"""
Single-MCU PeakNet1D binary sender/receiver.

Compares the single-MCU baseline with the distributed 5-MCU pipeline using:
- host TX-to-RX latency
- RX result interval
- result rate / throughput
- peak localization error

Protocol:
  MCU -> PC: b'R'
  PC -> MCU: b"ECG" + 256 float32 little-endian
  MCU -> PC: b"SCB" + uint16 length + 256 float32 scores

Example:
python3 ecg-sender-single-mcu-binary.py --port /dev/ttyUSB0 --baud 576000 --mode fast --duration 30
"""

import argparse
import time
import struct
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt


DEFAULT_PORT = "/dev/ttyUSB0"
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


def send_frame(ser, frame_f32, chunk_size=32, chunk_delay=0.002):
    """
    Send one ECG frame in small USB-CDC chunks.

    The STM32 USB CDC receive path is more reliable when the 1027-byte
    application packet is paced rather than written in one large burst.
    """
    frame_f32 = np.asarray(frame_f32, dtype=np.float32)
    if frame_f32.shape[0] != L:
        raise ValueError(f"frame length must be {L}, got {frame_f32.shape[0]}")

    packet = b"ECG" + frame_f32.astype("<f4", copy=False).tobytes(order="C")

    for i in range(0, len(packet), chunk_size):
        ser.write(packet[i:i + chunk_size])
        ser.flush()
        if chunk_delay > 0:
            time.sleep(chunk_delay)


class BinaryScoreReader:
    def __init__(self, expected_length=256):
        self.expected_length = expected_length
        self.buf = bytearray()

    def feed(self, data):
        if data:
            self.buf.extend(data)

    def poll(self):
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
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            scores = np.clip(scores, 0.0, 1.0)

        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--mode", choices=["fast", "realtime"], default="fast")
    parser.add_argument("--plot-window", type=float, default=10.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESH)
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="USB-CDC TX chunk size in bytes (default: 32).")
    parser.add_argument("--chunk-delay", type=float, default=0.002,
                        help="Delay between TX chunks in seconds (default: 0.002).")
    args = parser.parse_args()

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
    ready_permits = 0

    host_latency_times = []
    rx_arrival_times = []
    reader = BinaryScoreReader(expected_length=L)

    print(f"[host] opening serial {args.port} @ {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=0.0, write_timeout=2.0)

    time.sleep(1.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    host_t0 = time.perf_counter()

    print(f"[host] single-MCU baseline, mode={args.mode}, frames={len(starts)}")
    print("[host] waits for MCU READY byte 'R' before each TX")
    print("[host] reads binary SCORES packets from the same serial port")
    print(f"[host] ECG TX pacing: chunk={args.chunk_size} B, delay={args.chunk_delay*1000:.1f} ms")

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
        ax.set_title("Single-MCU PeakNet1D baseline")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
    else:
        fig = ax = score_line = pred_scatter = None

    last_plot_update = 0.0
    last_wait_print = 0.0

    while next_tx_index < len(starts) or pending:
        now = time.perf_counter()

        n_avail = ser.in_waiting
        if n_avail > 0:
            data = ser.read(n_avail)
            ready_count = data.count(b"R")
            if ready_count:
                ready_permits += ready_count
                print(f"[ready] +{ready_count}, permits={ready_permits}")
            reader.feed(data)

        scores = reader.poll()

        if scores is not None:
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

        can_send = next_tx_index < len(starts) and ready_permits > 0 and len(pending) == 0

        if can_send and args.mode == "realtime":
            start = starts[next_tx_index]
            target_time = host_t0 + start / FS
            if now < target_time:
                can_send = False

        if can_send:
            start = starts[next_tx_index]
            frame = normalize_frame(x[start:start + L])
            tx_t = time.perf_counter()
            send_frame(ser, frame, chunk_size=args.chunk_size, chunk_delay=args.chunk_delay)

            ready_permits -= 1
            next_tx_index += 1
            sent_count += 1

            pending.append({"start": start, "tx_index": sent_count, "tx_t": tx_t})
            print(f"[tx] {sent_count:04d}/{len(starts)} start={start} pending={len(pending)} permits={ready_permits}")

        if next_tx_index == 0 and ready_permits == 0 and not pending and (now - last_wait_print) > 2.0:
            last_wait_print = now
            print("[wait] waiting for initial READY byte R...")

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

            if recv_count > 0:
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

    ser.close()

    count_safe = count_all.copy()
    count_safe[count_safe == 0] = 1.0
    score_avg = score_all / count_safe

    peaks = score_to_peaks(score_avg, FS, thresh=args.threshold, refractory_s=REFRACTORY_S)
    true_idx = (true_beats * FS).astype(int)
    true_idx = true_idx[(true_idx >= 0) & (true_idx < n)]

    elapsed = time.perf_counter() - host_t0

    print("\n=== Single-MCU Sender Summary ===")
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
