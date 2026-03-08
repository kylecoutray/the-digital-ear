# Streaming HPSS (Harmonic-Percussive Source Separation)
#
# Circular buffer of kernel_h STFT frames, median filtering to get
# harmonic/percussive masks, overlap-add to reconstruct harmonic audio.
# ~414 KB constant memory, ~175ms latency at hop=512/sr=44100.
#
# Reference: Fitzgerald, "Harmonic/Percussive Separation using Median
# Filtering" (DAFx 2010)

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class HPSS:
    """Streaming HPSS. Feed audio via push(), get harmonic audio from
    pop_harmonic(), call flush() at end of stream."""

    n_fft: int = 2048
    hop: int = 512
    kernel_h: int = 31   # ~360ms at hop=512/44100
    kernel_p: int = 31   # ~660 Hz at n_fft=2048/44100
    power: float = 2.0

    # pre-computed
    _n_freq: int = field(init=False, repr=False)
    _half_k: int = field(init=False, repr=False)
    _window: np.ndarray = field(init=False, repr=False)
    _window_sq: np.ndarray = field(init=False, repr=False)

    # STFT circular buffer
    _stft_buf: np.ndarray = field(init=False, repr=False)   # (kernel_h, n_freq) complex64
    _mag_buf: np.ndarray = field(init=False, repr=False)     # (kernel_h, n_freq) float32
    _buf_count: int = field(init=False, repr=False, default=0)

    # audio accumulation buffer
    _audio_buf: np.ndarray = field(init=False, repr=False)
    _audio_fill: int = field(init=False, repr=False, default=0)

    # overlap-add output
    _ola_buf: np.ndarray = field(init=False, repr=False)
    _ola_norm: np.ndarray = field(init=False, repr=False)

    # output queue
    _output_queue: list = field(init=False, repr=False, default_factory=list)

    # percussive energy (for onset detection)
    _perc_energy_queue: list = field(init=False, repr=False, default_factory=list)

    # flushed flag
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
        """Feed audio samples. Accumulates into STFT frames internally."""
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
        """Pop hop-length harmonic audio blocks. May return 0+."""
        out = self._output_queue
        self._output_queue = []
        return out

    def pop_percussive_energy(self) -> list[float]:
        """Mean percussive magnitude per emitted frame (spikes = transients)."""
        out = self._perc_energy_queue
        self._perc_energy_queue = []
        return out

    def flush(self) -> list[np.ndarray]:
        """End of stream: emit remaining frames with partial median."""
        if self._flushed:
            return []
        self._flushed = True

        # remaining frames that never got to be center -- use partial median
        if self._buf_count <= self._half_k:
            # never had enough frames, emit all with partial median
            for i in range(self._buf_count):
                self._emit_frame_partial(i, self._buf_count)
        else:
            # emit the remaining half_k frames
            remaining = min(self._half_k, self._buf_count)
            for offset in range(remaining):
                # center position shifts forward
                center_rel = self._half_k + 1 + offset
                if center_rel >= self._buf_count:
                    center_rel = self._buf_count - 1
                center_pos = (self._buf_count - self._buf_count + center_rel) % self.kernel_h
                # use whatever frames we have
                n_avail = min(self._buf_count, self.kernel_h)
                self._emit_frame_partial(center_pos, n_avail)

        out = self._output_queue
        self._output_queue = []
        return out

    def _process_stft_frame(self) -> None:
        """Window, FFT, store in circular buffer, emit center if ready."""
        windowed = self._audio_buf * self._window
        X = np.fft.rfft(windowed).astype(np.complex64)
        mag = np.abs(X).astype(np.float32)

        # store in circular buffer
        pos = self._buf_count % self.kernel_h
        self._stft_buf[pos] = X
        self._mag_buf[pos] = mag
        self._buf_count += 1

        # emit center frame if buffer is full
        if self._buf_count >= self.kernel_h:
            # Center frame is half_k behind current
            center_pos = (self._buf_count - self._half_k - 1) % self.kernel_h
            self._emit_frame(center_pos)

    def _emit_frame(self, center_pos: int) -> None:
        """Compute masks and emit harmonic audio for one frame."""
        # time-axis median
        H = np.median(self._mag_buf, axis=0)  # (n_freq,)

        # freq-axis median for center frame
        center_mag = self._mag_buf[center_pos]
        P = self._median_filter_freq_single(center_mag)

        # wiener mask
        self._apply_mask_and_emit(center_pos, H, P)

    def _emit_frame_partial(self, center_pos: int, n_avail: int) -> None:
        """Like _emit_frame but with fewer than kernel_h frames available."""
        if n_avail <= 0:
            return
        # time-axis median with available frames
        if n_avail >= self.kernel_h:
            H = np.median(self._mag_buf, axis=0)
        else:
            # gather what we have
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
        """Wiener mask -> ISTFT -> overlap-add to output."""
        eps = np.float32(1e-10)
        Hp = np.power(H, self.power)
        Pp = np.power(P, self.power)
        mask = Hp / (Hp + Pp + eps)

        # percussive energy for onset detection
        perc_mask = Pp / (Hp + Pp + eps)
        center_mag = self._mag_buf[center_pos]
        perc_energy = float(np.mean(center_mag * perc_mask))
        self._perc_energy_queue.append(perc_energy)

        # apply mask
        masked_X = self._stft_buf[center_pos] * mask

        # ISTFT + overlap-add
        frame_audio = np.fft.irfft(masked_X, n=self.n_fft).real.astype(np.float32)
        frame_audio *= self._window

        # accumulate
        self._ola_buf += frame_audio
        self._ola_norm += self._window_sq

        # extract first hop samples
        out_block = self._ola_buf[:self.hop].copy()
        norm_block = self._ola_norm[:self.hop]
        valid = norm_block > 1e-8
        out_block[valid] /= norm_block[valid]

        # shift buffer
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
