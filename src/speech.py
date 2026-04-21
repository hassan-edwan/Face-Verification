"""
SpeechTranscriber — Non-blocking streaming speech-to-text.

Runs audio capture + transcription on a dedicated thread. The main thread reads
the current transcript state (partial + finalized lines) via a thread-safe
shared object. Never blocks the render loop.

Requires: vosk, sounddevice (optional runtime dependencies).
If not installed, the transcriber reports unavailable and all calls are no-ops.

Usage:
    transcriber = SpeechTranscriber(model_path="models/vosk-model-small-en-us")
    transcriber.start()       # begin listening
    ...
    state = transcriber.get_state()   # read from main thread each frame
    # state.partial  = "hello wor"    (still being spoken)
    # state.lines    = ["previous sentence finalized."]
    ...
    transcriber.stop()        # stop listening
"""

import json
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Callable

# ── Optional dependency check ─────────────────────────────────────────────────
_VOSK_AVAILABLE = False
_SD_AVAILABLE = False

try:
    import vosk
    _VOSK_AVAILABLE = True
except ImportError:
    pass

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    pass

SPEECH_AVAILABLE = _VOSK_AVAILABLE and _SD_AVAILABLE

# ── Transcription state (shared between audio thread and main thread) ────────

@dataclass
class TranscriptState:
    """Thread-safe snapshot of current transcription output."""
    is_listening:  bool = False       # True when microphone is active
    partial:       str  = ""          # current partial (in-progress) text
    lines:         List[str] = field(default_factory=list)  # finalized lines (most recent last)
    last_final_at: float = 0.0        # timestamp of last finalized line
    error:         str  = ""          # error message if something went wrong


# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000   # vosk models expect 16kHz
BLOCK_SIZE      = 4000    # ~250ms chunks at 16kHz (good balance: responsive but not too chatty)
MAX_LINES       = 20      # keep last N finalized lines in memory


class SpeechTranscriber:
    """Streaming speech-to-text on a dedicated thread.

    Audio capture (sounddevice) feeds chunks to a vosk recognizer.
    Partial results update every ~250ms; final results on silence/phrase boundary.
    """

    def __init__(self, model_path: str = ""):
        self._model_path = model_path
        self._lock = threading.Lock()
        self._state = TranscriptState()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_finalize: Optional[Callable[[str], None]] = None  # callback for finalized text
        self._model = None  # Cached vosk.Model — loaded once, reused across start/stop
        self._audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)  # ~16 s backlog

        # Check availability
        if not SPEECH_AVAILABLE:
            missing = []
            if not _VOSK_AVAILABLE:
                missing.append("vosk")
            if not _SD_AVAILABLE:
                missing.append("sounddevice")
            self._state.error = f"Speech unavailable (install: {', '.join(missing)})"

    @property
    def available(self) -> bool:
        return SPEECH_AVAILABLE

    def set_on_finalize(self, callback: Callable[[str], None]) -> None:
        """Set callback invoked on the audio thread when a phrase is finalized."""
        self._on_finalize = callback

    def start(self) -> bool:
        """Start listening. Returns False if dependencies unavailable or already running."""
        if not SPEECH_AVAILABLE:
            return False
        # Guard against double-start race: set _running BEFORE spawning the thread
        # so a rapid start()→start() can't create two audio threads on the same mic.
        if self._running or (self._thread and self._thread.is_alive()):
            return True
        self._running = True
        self._stop_event.clear()
        # Drain any stale audio from a prior session
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="SpeechTranscriber"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop listening."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._state.is_listening = False
            self._state.partial = ""

    def close(self) -> None:
        """Stop and release the Vosk model explicitly while the C module is
        still loaded. Fixes interpreter-shutdown race that manifested as
        `AttributeError: 'NoneType' object has no attribute 'vosk_model_free'`
        — Python GC would try to free the model after vosk's C bindings had
        already been unloaded."""
        try:
            self.stop()
        finally:
            # Drop the Python-side reference deterministically while vosk is
            # still importable. Guarded because close() may run multiple times.
            if self._model is not None:
                self._model = None
            # Drain any stale audio bytes so the queue itself is freed.
            try:
                while not self._audio_q.empty():
                    self._audio_q.get_nowait()
            except Exception:
                pass

    def get_state(self) -> TranscriptState:
        """Get a snapshot of the current transcript state. Non-blocking."""
        with self._lock:
            return TranscriptState(
                is_listening=self._state.is_listening,
                partial=self._state.partial,
                lines=list(self._state.lines),
                last_final_at=self._state.last_final_at,
                error=self._state.error,
            )

    def clear(self) -> None:
        """Clear all transcript history."""
        with self._lock:
            self._state.lines.clear()
            self._state.partial = ""

    # ── Internal: audio thread ────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """PortAudio callback — runs on the high-priority audio thread.

        MUST be fast: copy bytes and enqueue, nothing else. Dropping a chunk here
        (queue full) means the consumer is badly behind; we prefer dropping the
        newest frame over blocking the audio thread.
        """
        if status:
            # Log xruns but don't crash — PortAudio tells us about input overflow
            # via the `status` flags; we keep going.
            pass
        try:
            self._audio_q.put_nowait(bytes(indata))
        except queue.Full:
            # Consumer is stalled; drop newest chunk rather than stall audio thread.
            # Vosk will see a short gap and emit a phrase boundary — acceptable.
            pass

    def _ensure_model_loaded(self) -> None:
        """Load the Vosk model lazily on first start and cache it on the instance."""
        if self._model is not None:
            return
        if self._model_path:
            self._model = vosk.Model(self._model_path)
        else:
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(lang="en-us")

    def _listen_loop(self) -> None:
        """Runs on dedicated thread. Drains audio queue and feeds to vosk."""
        try:
            self._ensure_model_loaded()

            # Fresh recognizer per session so state doesn't carry across stop/start
            rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
            rec.SetWords(False)

            with self._lock:
                self._state.is_listening = True
                self._state.error = ""

            # Non-blocking capture: PortAudio pushes chunks to self._audio_q via
            # self._audio_callback. This loop only consumes — no blocking read.
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        data = self._audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            with self._lock:
                                self._state.lines.append(text)
                                if len(self._state.lines) > MAX_LINES:
                                    self._state.lines.pop(0)
                                self._state.partial = ""
                                self._state.last_final_at = time.time()
                            if self._on_finalize:
                                try:
                                    self._on_finalize(text)
                                except Exception:
                                    pass
                    else:
                        partial = json.loads(rec.PartialResult())
                        text = partial.get("partial", "").strip()
                        with self._lock:
                            self._state.partial = text

                # On clean stop, flush any final phrase sitting in the recognizer
                final = json.loads(rec.FinalResult()).get("text", "").strip()
                if final:
                    with self._lock:
                        self._state.lines.append(final)
                        if len(self._state.lines) > MAX_LINES:
                            self._state.lines.pop(0)
                        self._state.partial = ""
                        self._state.last_final_at = time.time()
                    if self._on_finalize:
                        try:
                            self._on_finalize(final)
                        except Exception:
                            pass

        except Exception as exc:
            with self._lock:
                self._state.error = str(exc)
            print(f"[Speech] Error: {exc}")
        finally:
            with self._lock:
                self._state.is_listening = False
            self._running = False
