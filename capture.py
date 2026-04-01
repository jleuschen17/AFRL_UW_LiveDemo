#!/usr/bin/env python3
"""
USRP + LNB Starlink SigMF Capture Script

- Frequency hopping: 10.7 GHz to 12.7 GHz in 125 MHz steps
- Capture duration: 0.5 s per frequency
- Sample rate: 0.5 MS/s (500 kS/s)
- Output: SigMF files (.sigmf-meta + .sigmf-data) in a new folder
"""

import uhd
import numpy as np
import time
from datetime import datetime, timezone
import os
import json
import math
from scipy import signal, constants
import gc


class StarlinkSigMFCapture:
    def __init__(
        self,
        usrp_args="type=b200",
        sample_rate=0.5e6,   # 0.5 MS/s
        gain=7,             # RX gain in dB
        lnb_lo=9.75e9 +0.2e6,       # LNB LO freq (Hz), e.g., 9.75 GHz
        output_root="./starlink_sigmf_captures",
    ):
        self.sample_rate = float(sample_rate)
        self.gain = float(gain)
        self.lnb_lo = float(lnb_lo)

        # Connect to USRP
        print(f"Connecting to USRP device: {usrp_args if usrp_args else 'default device'}")
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)

        # Configure RX chain
        self.usrp.set_rx_rate(self.sample_rate)
        self.usrp.set_rx_gain(self.gain)
        self.usrp.set_rx_antenna("RX2")

        print(f"USRP info:\n{self.usrp.get_pp_string()}")
        print(f"Configured sample rate: {self.usrp.get_rx_rate()/1e6:.3f} MS/s")
        print(f"Configured gain: {self.usrp.get_rx_gain():.1f} dB")
        print(f"LNB LO: {self.lnb_lo/1e9:.3f} GHz")

        # Create output directory
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.output_dir = os.path.join(output_root, f"starlink_sigmf_{ts}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Create RX streamer once (reused for every capture)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.rx_streamer = self.usrp.get_rx_stream(st_args)

        # Open a simple text log
        self.log_path = os.path.join(self.output_dir, "capture_log.txt")
        with open(self.log_path, "w") as log:
            log.write(f"Starlink SigMF Capture\n")
            log.write(f"Start time: {datetime.now()}\n")
            log.write(f"Sample rate: {self.sample_rate} Hz\n")
            log.write(f"Gain: {self.gain} dB\n")
            log.write(f"LNB LO: {self.lnb_lo} Hz\n\n")

    def set_frequency(self, target_freq_hz):
        """
        Set USRP frequency given desired RF frequency at the dish output (before LNB).

        target_freq_hz: desired RF center frequency [Hz], e.g., 10.7e9
        """
        target_freq_hz = float(target_freq_hz)
        if_freq = target_freq_hz - self.lnb_lo

        if if_freq <= 0:
            print(
                f"WARNING: target {target_freq_hz/1e9:.3f} GHz < LO "
                f"{self.lnb_lo/1e9:.3f} GHz (IF={if_freq/1e6:.2f} MHz)"
            )
            return False

        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(if_freq))
        actual_if = self.usrp.get_rx_freq()
        print(
            f"  Target RF: {target_freq_hz/1e9:.3f} GHz  |  "
            f"IF tune: {if_freq/1e6:.2f} MHz  (actual {actual_if/1e6:.2f} MHz)"
        )
        return True

    @staticmethod
    def _iso8601_utc_now():
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def capture_and_write_streaming(
        self, num_samples, center_freq_hz, round_idx, freq_idx,
        chunk_size=1_000_000,
        nfft=1024, noverlap=512, k=10, DC_width=5000,
    ):
        """
        Two-phase capture:
          Phase 1 — Stream IQ to disk as fast as possible (recv + write only).
          Phase 2 — Read the file back in chunks and compress offline.

        The receive loop does NO processing beyond a running power sum,
        so it can keep up with the USRP and avoid overflows.

        Returns (meta_path, data_path, total_received, power_dbm).
        """
        num_samples = int(num_samples)
        center_freq_hz = float(center_freq_hz)

        # ---- Build file paths ----
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base_name = f"r{round_idx:03d}_f{center_freq_hz/1e9:.3f}GHz_{ts}"
        data_path = os.path.join(self.output_dir, base_name + ".sigmf-data")
        meta_path = os.path.join(self.output_dir, base_name + ".sigmf-meta")
        comp_data_path = os.path.join(self.output_dir, base_name + ".txt")

        # ---- Prepare streaming ----
        recv_buffer = np.zeros(chunk_size, dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        overflow_count = 0

        # Use continuous streaming (no num_samps limit)
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        total_received = 0
        power_accum = 0.0

        # ============================================================
        # PHASE 1: Stream to disk (recv + write ONLY — no compression)
        # ============================================================
        print("    Phase 1: Streaming to disk...")
        with open(data_path, "wb") as data_file:
            while total_received < num_samples:
                remaining = num_samples - total_received
                request_len = min(chunk_size, remaining)

                samps = self.rx_streamer.recv(
                    recv_buffer[:request_len], metadata
                )

                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                        overflow_count += 1
                        continue
                    else:
                        print(f"    RX error: {metadata.strerror()}")
                        break

                if samps == 0:
                    continue

                chunk = recv_buffer[:samps]

                # Write raw IQ to disk immediately
                chunk.tofile(data_file)

                # Running power accumulator (cheap — just abs² + sum)
                power_accum += np.sum(np.abs(chunk) ** 2)

                total_received += samps

                # Progress every ~10M samples
                if (total_received // 10_000_000) != ((total_received - samps) // 10_000_000):
                    pct = 100.0 * total_received / num_samples
                    print(f"    ... {total_received:,}/{num_samples:,} samples ({pct:.1f}%)")

        # ---- Stop streaming ----
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stop_cmd)

        # Flush any residual samples out of the streamer
        flush_buf = np.zeros(chunk_size, dtype=np.complex64)
        flush_md = uhd.types.RXMetadata()
        while True:
            n = self.rx_streamer.recv(flush_buf, flush_md, timeout=0.1)
            if n == 0:
                break

        if overflow_count > 0:
            print(f"    WARNING: {overflow_count} overflows during capture")

        print(f"    Phase 1 complete: {total_received:,} samples written to disk")

        # ============================================================
        # PHASE 2: Read back from disk and compress (offline, no time pressure)
        # ============================================================
        print("    Phase 2: Compressing from disk...")
        all_compressed = []
        with open(data_path, "rb") as data_file:
            while True:
                raw = np.fromfile(data_file, dtype=np.complex64, count=chunk_size)
                if len(raw) == 0:
                    break
                if len(raw) >= nfft:
                    comp = self.compress_samples(
                        raw, self.sample_rate, center_freq=center_freq_hz,
                        k=k, nfft=nfft, noverlap=noverlap, DC_width=DC_width,
                    )
                    all_compressed.append(comp)
                del raw

        # Write compressed data
        if all_compressed:
            comp_all = np.vstack(all_compressed)
            np.savetxt(comp_data_path, comp_all, fmt="%.10g", delimiter=" ")
            del comp_all
        del all_compressed
        gc.collect()

        print("    Phase 2 complete: compressed data written")

        # ---- Write SigMF metadata ----
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": float(self.sample_rate),
                "core:version": "1.0.0",
                "core:description": (
                    "Starlink Ku-band frequency-hopping capture "
                    "(USRP + LNB, one capture per file)"
                ),
                "core:author": "starlink_capture_script",
                "core:recorder": "USRP + LNB",
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": center_freq_hz,
                    "core:datetime": self._iso8601_utc_now(),
                }
            ],
            "annotations": [],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # ---- Power estimate ----
        power_dbm = (
            10 * np.log10(power_accum / max(total_received, 1) + 1e-12) + 30
        )

        return meta_path, data_path, total_received, power_dbm

    def frequency_hopping_capture(
        self,
        start_freq=10.7e9,
        end_freq=12.7e9,
        freq_step=125e6,
        capture_duration=0.5,
        total_duration=None,
        frequencies=None,
    ):
        """
        Frequency-hopping capture.

        - start_freq, end_freq, freq_step in Hz (RF, before LNB)
        - capture_duration in seconds (per frequency)
        - total_duration: if None, do a single pass; otherwise repeat rounds
        """
        start_freq = float(start_freq)
        end_freq = float(end_freq)
        freq_step = float(freq_step)
        capture_duration = float(capture_duration)

        # Build frequency list (inclusive of end_freq)
        if frequencies is not None: 
            freqs = [float(f) for f in frequencies]
        else:
            num_steps = int(math.floor((end_freq - start_freq) / freq_step))
            freqs = [start_freq + i * freq_step for i in range(num_steps + 1)]
        num_freqs = len(freqs)

        num_samples = int(self.sample_rate * capture_duration)

        time_per_round = num_freqs * capture_duration
        if total_duration is None:
            num_rounds = 1
            total_duration = time_per_round
        else:
            total_duration = float(total_duration)
            num_rounds = int(math.ceil(total_duration / time_per_round))

        print("\n=== Frequency-Hopping SigMF Capture ===")
        print(f"RF range: {start_freq/1e9:.3f} – {end_freq/1e9:.3f} GHz")
        print(f"Step: {freq_step/1e6:.0f} MHz  |  {num_freqs} freqs per round")
        print(f"Sample rate: {self.sample_rate/1e6:.3f} MS/s")
        print(f"Capture duration per freq: {capture_duration:.3f} s")
        print(f"Samples per freq: {num_samples}")
        print(f"Time per round: {time_per_round:.2f} s")
        print(f"Number of rounds: {num_rounds}")
        print(f"Estimated total capture time: {num_rounds * time_per_round:.2f} s\n")

        total_captures = 0

        with open(self.log_path, "a") as log:
            for r in range(1, num_rounds + 1):
                if num_rounds > 1:
                    print(f"\n===== ROUND {r}/{num_rounds} =====\n")
                    log.write(f"--- Round {r} ---\n")

                for i, rf in enumerate(freqs, start=1):
                    total_captures += 1
                    print(
                        f"[Round {r}/{num_rounds}] "
                        f"[Freq {i}/{num_freqs}] RF={rf/1e9:.3f} GHz"
                    )

                    if not self.set_frequency(rf):
                        log.write(
                            f"{datetime.now()} | Round {r} | RF {rf/1e9:.3f} GHz | "
                            f"FAIL: invalid IF\n"
                        )
                        log.flush()
                        continue

                    # Give LO + PLL a moment to settle
                    time.sleep(0.1)

                    start_t = time.time()
                    meta_path, data_path, num_recv, power_dbm = \
                        self.capture_and_write_streaming(
                            num_samples=num_samples,
                            center_freq_hz=rf,
                            round_idx=r,
                            freq_idx=i,
                        )
                    elapsed = time.time() - start_t

                    print(
                        f"  Captured {num_recv} samples "
                        f"({elapsed:.2f} s), est power {power_dbm:.2f} dBm"
                    )
                    print(f"  -> {os.path.basename(meta_path)}")
                    print(f"  -> {os.path.basename(data_path)}\n")

                    log.write(
                        f"{datetime.now()} | Round {r} | RF {rf/1e9:.3f} GHz | "
                        f"{num_recv} samples | {power_dbm:.2f} dBm | "
                        f"{elapsed:.2f} s | {os.path.basename(meta_path)}\n"
                    )
                    log.flush()

        print("\n========================================")
        print("Capture complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total captures: {total_captures}")
        print("========================================\n")


    # Compress the data to store only the k strongest points
    def compress_samples(self,
        samples,
        sample_rate,
        center_freq,
        k = 10,
        nfft = 1024,
        noverlap = 512,
        DC_width = 5000,
    ):
        # Basic setup
        samples = np.array(samples)
        dataLen = len(samples)

        # Create STFT from the raw data
        f, t, STFT = signal.stft(samples, fs=sample_rate, nperseg=nfft, noverlap=noverlap, return_onesided=False)

        # Shift in-place (no extra copy)
        STFT = np.fft.fftshift(STFT, axes=0)
        f = np.fft.fftshift(f)
    
        # Determine how many frames of data there are
        numFrames = 1 + ((dataLen - nfft) // (nfft//2))

        # Initialize output variables
        output_vals = np.zeros((numFrames,k),dtype=complex)
        output_freqs = np.zeros((numFrames,k))

        # Identify the indexes of the DC values to remove 
        kill_inds = np.where((np.abs(f) <= DC_width))

        c = constants.c


        #Store the largest k values 
        for i in range(numFrames):

            working_row = np.abs(STFT[:,i])

            #Ignore the DC Band
            working_row[kill_inds] = 0
            
            #Find the max
            large_inds = np.argpartition(working_row, -k)[-k:]
            large_inds = large_inds[np.argsort(f[large_inds])]

            large_vals = STFT[large_inds,i]

            sat_freq = f[np.argmax(abs(working_row))]

            relVels = c*f[large_inds]/center_freq

            output_vals[i,:] = large_vals
            output_freqs[i,:] = f[large_inds]


        # Merge the frequency data with the complex values
        output_file = np.column_stack((output_freqs,output_vals))

        return output_file



    def __del__(self):
        print("Closing USRP connection")

    



    


def main():
    # ---- User-configurable bits ----
    USRP_ARGS = "type=b200"       # or "" or "addr=..." etc
    SAMPLE_RATE = 1.0e6           # 0.5 MS/s
    GAIN = 40                     # dB
    LNB_LO = 9.75e9 + 0.2e6               # Hz (typical Ku-band LNB)
    OUTPUT_ROOT = "/home/mowerj/workarea_starlink/uhd/AFRL_tests/automatedHopping"

    START_FREQ = 10.7e9           # 10.7 GHz
    END_FREQ = 12.7e9             # 12.7 GHz
    FREQ_STEP = 125e6             # 125 MHz
    CAPTURE_DURATION = 30        # seconds per freq
    TOTAL_DURATION = 30*2*2 #------------
    SPECIFIC_FREQS =[11.575e9, 12.325e9]
    # SPECIFIC_FREQS = [11.575e9]

    try:
        cap = StarlinkSigMFCapture(
            usrp_args=USRP_ARGS,
            sample_rate=SAMPLE_RATE,
            gain=GAIN,
            lnb_lo=LNB_LO,
            output_root=OUTPUT_ROOT,
        )

        cap.frequency_hopping_capture(
            start_freq=START_FREQ,
            end_freq=END_FREQ,
            freq_step=FREQ_STEP,
            capture_duration=CAPTURE_DURATION,
            total_duration=TOTAL_DURATION,
            frequencies=SPECIFIC_FREQS,
            # frequencies=None

        )

    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()