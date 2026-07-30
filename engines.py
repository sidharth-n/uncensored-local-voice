"""Swappable engine slots for the voice pipeline.

Why this exists: every component in the pipeline (STT, TTS, end-of-turn) is a
model choice we want to A/B on real hardware, not a permanent decision. Each
slot is a small protocol plus one adapter per backend, selected by an env var.

The defaults reproduce the original hard-wired pipeline exactly — Moonshine STT,
Kokoro TTS, no semantic turn detection — so an unset environment behaves
identically to before this module existed.

    STT_ENGINE=moonshine   (default)
    TTS_ENGINE=kokoro      (default)
    TURN_DETECTOR=off      (default)
"""

from __future__ import annotations

import os
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

SR = 16000  # pipeline-wide sample rate; TTS adapters resample to this upstream


# ─────────────────────────────── protocols ────────────────────────────


@runtime_checkable
class SttEngine(Protocol):
    """Utterance-at-a-time speech to text.

    The pipeline buffers a whole utterance behind VAD before transcribing, so
    engines only need a batch call — not incremental partial hypotheses.
    """

    name: str

    def transcribe(self, audio_f32: np.ndarray) -> str: ...
    def warmup(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class TtsEngine(Protocol):
    """Sentence-at-a-time speech synthesis.

    `stream` yields float32 mono chunks at `sample_rate`. The player resamples
    to 16 kHz for both the speaker and the AEC reverse stream, so adapters
    report their native rate rather than resampling themselves.
    """

    name: str
    sample_rate: int

    def stream(self, sentence: str) -> Iterator[np.ndarray]: ...
    def warmup(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class TurnDetector(Protocol):
    """Decides whether a VAD-delimited utterance is actually finished.

    Silero tells us the user stopped making sound; it cannot tell us whether
    they finished their thought. A semantic detector answers the second
    question, which is what stops the agent cutting people off mid-sentence.
    """

    name: str

    def is_complete(self, audio_f32: np.ndarray) -> bool: ...
    def close(self) -> None: ...


# ─────────────────────────────── STT adapters ─────────────────────────


class MoonshineStt:
    """Moonshine via the `moonshine_voice` package. English only.

    MEDIUM_STREAMING is more accurate than BASE and STT sits well inside our
    latency budget, so we spend the extra compute on quality. The _STREAMING
    variant works fine for one-shot `transcribe_without_streaming` calls.
    """

    name = "moonshine"

    def __init__(self, language: str = "en", arch: str | None = None) -> None:
        from moonshine_voice import get_model_for_language
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber

        resolved = getattr(ModelArch, arch) if arch else ModelArch.MEDIUM_STREAMING
        path, model_arch = get_model_for_language(language, resolved)
        self._t = Transcriber(model_path=str(path), model_arch=model_arch)
        self._t.start()

    def transcribe(self, audio_f32: np.ndarray) -> str:
        transcript = self._t.transcribe_without_streaming(
            audio_f32.tolist(), sample_rate=SR
        )
        parts = []
        for line in getattr(transcript, "lines", []) or []:
            text = getattr(line, "text", "") or ""
            if text.strip():
                parts.append(text.strip())
        return " ".join(parts).strip()

    def warmup(self) -> None:
        self.transcribe(np.zeros(SR, dtype=np.float32))

    def close(self) -> None:
        try:
            self._t.stop()
            self._t.close()
        except Exception:
            pass


# ─────────────────────────────── TTS adapters ─────────────────────────


class KokoroTts:
    """Kokoro 82M via `kokoro_mlx`. Fast, English-centric, 24 kHz out.

    Pinned to kokoro-mlx 0.1.1 by the interpreter: 0.1.2 requires Python <3.13
    and this venv is 3.13. That ceiling is one reason the TTS slot exists.
    """

    name = "kokoro"

    def __init__(self, voice: str | None = None) -> None:
        from kokoro_mlx import DEFAULT_VOICE, KokoroTTS

        self._tts = KokoroTTS.from_pretrained()
        self._voice = voice or DEFAULT_VOICE
        self.sample_rate = int(self._tts.SAMPLE_RATE)  # 24000

    def stream(self, sentence: str) -> Iterator[np.ndarray]:
        for chunk in self._tts.generate_stream(sentence, voice=self._voice):
            yield np.asarray(chunk, dtype=np.float32).flatten()

    def warmup(self) -> None:
        for _ in self.stream("Hi."):
            pass

    def close(self) -> None:
        pass


# ─────────────────────────── turn detectors ───────────────────────────


class NoTurnDetector:
    """The original behaviour: trust Silero's end-of-speech and reply.

    Kept as a named engine rather than a None check so the pipeline has one
    code path, and so `TURN_DETECTOR=off` stays a first-class, testable choice.
    """

    name = "off"

    def is_complete(self, audio_f32: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        pass


# ─────────────────────────────── registry ─────────────────────────────


def build_stt() -> SttEngine:
    choice = os.environ.get("STT_ENGINE", "moonshine").strip().lower()
    if choice == "moonshine":
        return MoonshineStt(
            language=os.environ.get("STT_LANGUAGE", "en"),
            arch=os.environ.get("MOONSHINE_ARCH") or None,
        )
    raise SystemExit(
        f"unknown STT_ENGINE={choice!r}. available: moonshine"
    )


def build_tts() -> TtsEngine:
    choice = os.environ.get("TTS_ENGINE", "kokoro").strip().lower()
    if choice == "kokoro":
        return KokoroTts(voice=os.environ.get("TTS_VOICE") or None)
    raise SystemExit(
        f"unknown TTS_ENGINE={choice!r}. available: kokoro"
    )


def build_turn_detector() -> TurnDetector:
    choice = os.environ.get("TURN_DETECTOR", "off").strip().lower()
    if choice in ("off", "none", "0"):
        return NoTurnDetector()
    raise SystemExit(
        f"unknown TURN_DETECTOR={choice!r}. available: off"
    )
