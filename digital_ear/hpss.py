# Streaming Harmonic-Percussive Source Separation (HPSS)
#
# Separates audio into harmonic (tonal) and percussive (transient) components
# using median filtering on a sliding window of STFT frames.
#
# Streaming design: maintains a circular buffer of `kernel_h` (31) STFT frames.
# When the buffer is full, computes the harmonic/percussive masks for the center
# frame and emits the reconstructed harmonic audio via overlap-add.
#
# Memory: ~414 KB constant regardless of audio length (vs ~300 MB batch).
# Latency: kernel_h // 2 = 15 frames (~175 ms at hop=512/sr=44100).
#
# Reference: Fitzgerald, "Harmonic/Percussive Separation using Median
# Filtering" (DAFx 2010)

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class HPSS:
    """
    Streaming Harmonic-Percussive Source Separation.

    Feed audio blocks via push(), retrieve harmonic audio via pop_harmonic().
    At end of stream, call flush() to emit remaining frames.

    State: circular buffer of kernel_h STFT frames (~414 KB).
    """

    n_fft: int = 2048
    hop: int = 512
    kernel_h: int = 31   # ~360ms at hop=512/44100
    kernel_p: int = 31   # ~660 Hz at n_fft=2048/44100
    power: float = 2.0

    # Pre-computed
    _n_freq: int = field(init=False, repr=False)
    _half_k: int = field(init=False, repr=False)
    _window: np.ndarray = field(init=False, repr=False)
    _window_sq: np.ndarray = field(init=False, repr=False)

    # Circular STFT buffer
    _stft_buf: np.ndarray = field(init=False, repr=False)   # (kernel_h, n_freq) complex64
    _mag_buf: np.ndarray = field(init=False, repr=False)     # (kernel_h, n_freq) float32
    _buf_count: int = field(init=False, repr=False, default=0)

    # Audio accumulation buffer (samples → STFT frames)
    _audio_buf: np.ndarray = field(init=False, repr=False)
    _audio_fill: int = field(init=False, repr=False, default=0)

    # Overlap-add output buffer
    _ola_buf: np.ndarray = field(init=False, repr=False)
    _ola_norm: np.ndarray = field(init=False, repr=False)

    # Output queue
    _output_queue: list = field(init=False, repr=False, default_factory=list)

    # Track whether we've flushed
    _flushed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._n_freq = self.n_fft // 2 + 1
        self._half_k = self.kernel_h // 2
        self._window = np.hanning(self.n_fft).astype(np.float32)
        self._window_sq = self._window ** 2

        self._stft_buf = np.zeros((self.kernel_h, self._n_freq), dtype=np.complex64)
        self._mag_buf = np.zeros((self.kernel_h, self._n_freq), dtype=np.float32)

        self._audio_buf = np.zeros(self.n_fft, dtype=np.float32)
        self._ola_buf = np.zeros(self.n_fft, dtype=np.float32)
        self._ola_norm = np.zeros(self.n_fft, dtype=np.float32)

    def push(self, audio_block: np.ndarray) -> None:
        """Feed up to 2048 audio samples. Internally accumulates into
        STFT frames and processes when ready."""
        x = audio_block.astype(np.float32, copy=False)
        idx = 0
        while idx < len(x):
            space = self.n_fft - self._audio_fill
            take = min(space, len(x) - idx)
            self._audio_buf[self._audio_fill:self._audio_fill + take] = x[idx:idx + take]
            self._audio_fill += take
            idx += take

            if self._audio_fill == self.n_fft:
                self._process_stft_frame()
                # Shift by hop (keep overlap)
                remain = self.n_fft - self.hop
                self._audio_buf[:remain] = self._audio_buf[self.hop:]
                self._audio_buf[remain:] = 0.0
                self._audio_fill = remain

    def pop_harmonic(self) -> list[np.ndarray]:
        """Return list of hop-length harmonic audio blocks ready for
        downstream consumption. May return 0, 1, or more blocks."""
        out = self._output_queue
        self._output_queue = []
        return out

    def flush(self) -> list[np.ndarray]:
        """Process remaining buffered frames at end of stream.
        Uses partial median for frames that haven't been emitted yet."""
        if self._flushed:
            return []
        self._flushed = True

        # How many frames are in the buffer that haven't been emitted as center?
        # We've emitted center frames for buf_count - kernel_h frames (approximately).
        # The last half_k frames are still waiting to be center frames.
        # Process them with partial median (using available frames).
        if self._buf_count <= self._half_k:
            # Never had enough frames — emit all with partial median
            for i in range(self._buf_count):
                self._emit_frame_partial(i, self._buf_count)
        else:
            # Emit the remaining half_k frames that were waiting
            remaining = min(self._half_k, self._buf_count)
            for offset in range(remaining):
                # Center position shifts forward from the last emitted center
                center_rel = self._half_k + 1 + offset
                if center_rel >= self._buf_count:
                    center_rel = self._buf_count - 1
                center_pos = (self._buf_count - self._buf_count + center_rel) % self.kernel_h
                # Use all available frames for median
                n_avail = min(self._buf_count, self.kernel_h)
                self._emit_frame_partial(center_pos, n_avail)

        out = self._output_queue
        self._output_queue = []
        return out

    def _process_stft_frame(self) -> None:
        """Process one n_fft-length audio frame into the STFT buffer."""
        # Window and FFT
        windowed = self._audio_buf * self._window
        X = np.fft.rfft(windowed).astype(np.complex64)
        mag = np.abs(X).astype(np.float32)

        # Store in circular buffer
        pos = self._buf_count % self.kernel_h
        self._stft_buf[pos] = X
        self._mag_buf[pos] = mag
        self._buf_count += 1

        # Can we emit the center frame?
        if self._buf_count >= self.kernel_h:
            # Center frame is half_k behind current
            center_pos = (self._buf_count - self._half_k - 1) % self.kernel_h
            self._emit_frame(center_pos)

    def _emit_frame(self, center_pos: int) -> None:
        """Compute masks and emit harmonic audio for a frame using full kernel."""
        # Time-axis median: median of all kernel_h frames at each freq bin
        H = np.median(self._mag_buf, axis=0)  # (n_freq,)

        # Freq-axis median for center frame
        center_mag = self._mag_buf[center_pos]
        P = self._median_filter_freq_single(center_mag)

        # Wiener mask
        self._apply_mask_and_emit(center_pos, H, P)

    def _emit_frame_partial(self, center_pos: int, n_avail: int) -> None:
        """Emit a frame using partial median (fewer than kernel_h frames)."""
        if n_avail <= 0:
            return
        # Use only the available frames for time-axis median
        if n_avail >= self.kernel_h:
            H = np.median(self._mag_buf, axis=0)
        else:
            # Gather available frames
            indices = [(self._buf_count - n_avail + i) % self.kernel_h
                       for i in range(n_avail)]
            avail_mags = self._mag_buf[indices]
            H = np.median(avail_mags, axis=0)

        center_mag = self._mag_buf[center_pos]
        P = self._median_filter_freq_single(center_mag)
        self._apply_mask_and_emit(center_pos, H, P)

    def _apply_mask_and_emit(
        self, center_pos: int, H: np.ndarray, P: np.ndarray
    ) -> None:
        """Apply Wiener mask to center STFT frame and overlap-add to output."""
        eps = np.float32(1e-10)
        Hp = np.power(H, self.power)
        Pp = np.power(P, self.power)
        mask = Hp / (Hp + Pp + eps)

        # Apply mask to complex STFT frame
        masked_X = self._stft_buf[center_pos] * mask

        # ISTFT: irfft + overlap-add
        frame_audio = np.fft.irfft(masked_X, n=self.n_fft).real.astype(np.float32)
        frame_audio *= self._window

        # Overlap-add
        self._ola_buf += frame_audio
        self._ola_norm += self._window_sq

        # Extract the first hop samples as output
        out_block = self._ola_buf[:self.hop].copy()
        norm_block = self._ola_norm[:self.hop]
        valid = norm_block > 1e-8
        out_block[valid] /= norm_block[valid]

        # Shift overlap-add buffer
        remain = self.n_fft - self.hop
        self._ola_buf[:remain] = self._ola_buf[self.hop:]
        self._ola_buf[remain:] = 0.0
        self._ola_norm[:remain] = self._ola_norm[self.hop:]
        self._ola_norm[remain:] = 0.0

        self._output_queue.append(out_block)

    def _median_filter_freq_single(self, mag: np.ndarray) -> np.ndarray:
        """Median filter along frequency axis for a single frame."""
        half = self.kernel_p // 2
        padded = np.pad(mag, (half, half), mode='reflect')
        from numpy.lib.stride_tricks import sliding_window_view
        windowed = sliding_window_view(padded, self.kernel_p)  # (n_freq, kernel_p)
        return np.median(windowed, axis=1).astype(np.float32)
