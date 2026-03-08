from __future__ import annotations

import json
import subprocess
import numpy as np
import typing

# decode streaming helper using ffmpeg. We use subprocess to call ffmpeg and stream raw PCM audio data from it.

def probe_sample_rate(path: str) -> int | None:
    """Returns input sample rate via ffprobe, or None on failure."""
    try:
        cp = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate",
                "-of", "json",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(cp.stdout)
        sr = data["streams"][0].get("sample_rate", None)
        return int(sr) if sr is not None else None
    except Exception:
        return None

def stream_m4a_blocks_ffmpeg(
    in_path: str,
    block_size: int,
    target_sr: int = 44100,) -> tuple[int, "typing.Iterator[np.ndarray]"]:
    """Streams decoded mono float32 PCM blocks from ffmpeg.
    Returns (output_sample_rate, block_iterator)."""


    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", in_path,
        "-f", "s16le",          # raw 16-bit PCM
        "-acodec", "pcm_s16le",
        "-ac", "1",             # mono -> "audio channel -> 1" forces to mono
        "-ar", str(target_sr),  # resample to known SR
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.stdout is None:
        raise RuntimeError("Failed to open ffmpeg stdout pipe.")

    bytes_per_sample = 2  # 16-bit PCM
    read_bytes = block_size * bytes_per_sample

    def _iter_blocks() -> typing.Iterator[np.ndarray]:
        try:
            while True:
                raw = proc.stdout.read(read_bytes)
                if not raw:
                    break

                # raw length may be < read_bytes at end of file
                # Ensure even number of bytes for int16 decode
                if (len(raw) % 2) == 1:
                    raw = raw[:-1]

                x_i16 = np.frombuffer(raw, dtype=np.int16)
                x = (x_i16.astype(np.float32) / 32768.0)  # normalize to [-1, 1]

                # sanity check: never exceed block_size
                if x.shape[0] > block_size:
                    raise RuntimeError(f"Internal error: block len {x.shape[0]} > {block_size}")

                yield x
        finally:
            # Clean up ffmpeg
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            proc.wait(timeout=5)

    return target_sr, _iter_blocks()