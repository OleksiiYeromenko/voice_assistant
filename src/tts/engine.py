"""Streaming TTS using Piper — synthesize per-sentence, play immediately.

Key idea: as the LLM streams tokens, we buffer until we hit a sentence boundary
(sentence-ending punctuation), then synthesize that sentence with Piper and play it via aplay.
This means the user hears audio while the LLM is still generating.

Audio is piped directly from Piper to aplay in memory — no WAV files written to disk.
"""

import io
import logging
import math
import os
import random
import re
import struct
import subprocess
import threading
import time
import wave
from collections.abc import Generator
from pathlib import Path

log = logging.getLogger(__name__)

SENTENCE_RE = re.compile(r"\.(?!\d)\s+|[!?;]\s+")

_AMIXER_CONTROLS = ("PCM", "Master", "Speaker", "Headphone")


def _parse_card_number(aplay_device: str) -> str | None:
    m = re.match(r"(?:plug)?hw:(\d+)", aplay_device)
    return m.group(1) if m else None


def _apply_amixer_volume(card: str, level: int) -> None:
    for ctrl in _AMIXER_CONTROLS:
        try:
            r = subprocess.run(
                ["amixer", "-c", card, "sset", ctrl, f"{level}%"],
                capture_output=True,
                timeout=2,
            )
            if r.returncode == 0:
                log.info("TTS volume: set '%s' on card %s to %d%%", ctrl, card, level)
                return
        except FileNotFoundError:
            log.warning("amixer not found — TTS volume setting skipped")
            return
        except subprocess.TimeoutExpired:
            continue
    log.warning("TTS volume: no working control found on card %s (tried %s)", card, _AMIXER_CONTROLS)


class _DelayedSound:
    """Plays a WAV after a configurable delay in a background thread. Cancellable.

    Used as pre_proc in stream_speak() so the thinking sound starts a beat after
    transcription (feels natural) while still blocking TTS until it finishes.
    """

    def __init__(self, wav_data: bytes, aplay_device: str, delay_s: float):
        self._wav_data = wav_data
        self._aplay_device = aplay_device
        self._delay_s = delay_s
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="thinking-sound")
        self._thread.start()

    def _run(self):
        # Wait for the delay; threading.Event.wait returns True if cancelled early
        if self._cancel.wait(timeout=self._delay_s):
            return
        try:
            proc = subprocess.Popen(
                ["aplay", "-D", self._aplay_device],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(self._wav_data)
            proc.stdin.close()
            self._proc = proc
            proc.wait()
        except Exception:
            pass

    def wait(self, poll_interval: float = 0.05):
        """Block until delay+playback thread finishes."""
        while self._thread.is_alive():
            time.sleep(poll_interval)

    def cancel(self):
        """Stop delay or kill aplay if already started."""
        self._cancel.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                pass


class TTSEngine:
    def __init__(
        self,
        voice: str = "./voices/en_US-hfc_male-medium.onnx",
        aplay_device: str = "plughw:0,0",
        sounds_dir: str = "./sounds",
        volume: int = 80,
    ):
        self.voice = voice
        self.aplay_device = aplay_device
        self._beep_wav: bytes | None = None
        self._startup_sounds: list[Path] = []
        self._greeting_sounds: list[Path] = []
        self._thinking_sounds: list[Path] = []
        self._load_sounds(sounds_dir)
        self._apply_volume(max(0, min(100, int(volume))))

    # ------------------------------------------------------------------
    # Sound bank (pre-recorded WAV files)
    # ------------------------------------------------------------------
    def _load_sounds(self, sounds_dir: str):
        """Discover pre-recorded WAV files and store their paths."""
        base = Path(sounds_dir)
        mapping = {
            "startup": self._startup_sounds,
            "greeting": self._greeting_sounds,
            "thinking": self._thinking_sounds,
        }
        for category, target_list in mapping.items():
            category_dir = base / category
            if not category_dir.is_dir():
                continue
            target_list.extend(sorted(category_dir.glob("*.wav")))
            if target_list:
                log.info(f"Found {len(target_list)} {category} sound(s) in {category_dir}")

    def _apply_volume(self, level: int) -> None:
        card = _parse_card_number(self.aplay_device)
        if card is None:
            log.debug("aplay_device '%s' is not hw:/plughw: — skipping amixer", self.aplay_device)
            return
        _apply_amixer_volume(card, level)

    # ------------------------------------------------------------------
    # Acknowledgment beep
    # ------------------------------------------------------------------
    def _ensure_beep(self):
        """Generate a short beep WAV in memory."""
        if self._beep_wav is not None:
            return

        sample_rate = 22050
        duration = 0.15  # 150 ms
        freq = 880  # A5 note
        n_samples = int(sample_rate * duration)

        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(n_samples):
                t = i / sample_rate
                # Fade in/out to avoid clicks
                envelope = min(t / 0.01, 1.0) * min((duration - t) / 0.01, 1.0)
                sample = int(16000 * envelope * math.sin(2 * math.pi * freq * t))
                wf.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

        self._beep_wav = buf.getvalue()
        log.debug("Beep WAV generated in memory")

    def play_beep(self) -> subprocess.Popen | None:
        """Play acknowledgment beep (non-blocking). Returns Popen or None."""
        self._ensure_beep()
        try:
            proc = subprocess.Popen(
                ["aplay", "-D", self.aplay_device],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(self._beep_wav)
            proc.stdin.close()
            return proc
        except Exception as e:
            log.warning(f"Beep failed: {e}")
            return None

    def play_startup(self) -> None:
        """Play startup sound (blocking). Falls back to no-op if no sounds found."""
        if not self._startup_sounds:
            return
        proc = self._start_playback(self._startup_sounds[0].read_bytes())
        self._wait_for_playback(proc)

    def play_greeting(self) -> subprocess.Popen | None:
        """Play a random greeting sound (non-blocking). Falls back to beep if none found."""
        if not self._greeting_sounds:
            return self.play_beep()
        return self._start_playback(random.choice(self._greeting_sounds).read_bytes())

    def play_thinking(self, delay_s: float = 0.7) -> "_DelayedSound | subprocess.Popen | None":
        """Play a random thinking/processing sound. Returns None if none found.

        delay_s: seconds to wait before starting playback (default 0.7).
                 The sound runs in a background thread so the LLM starts immediately.
                 Pass delay_s=0 for immediate playback (returns Popen).
        """
        if not self._thinking_sounds:
            return None
        wav = random.choice(self._thinking_sounds).read_bytes()
        if delay_s <= 0:
            return self._start_playback(wav)
        return _DelayedSound(wav, self.aplay_device, delay_s)

    # ------------------------------------------------------------------
    # Core TTS
    # ------------------------------------------------------------------
    def synthesize(self, text: str) -> tuple[bytes, float]:
        """Synthesize text to WAV bytes in memory. Returns (wav_bytes, synth_time_s).

        Uses a temp file because piper's Python package requires --output_file
        (without it, piper tries to play audio itself via ffplay).
        The temp file is deleted immediately after reading.
        """
        import tempfile

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            start = time.perf_counter()
            result = subprocess.run(
                ["piper", "--model", self.voice, "--output_file", tmp_path],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            synth_time = time.perf_counter() - start

            if result.returncode != 0:
                log.error(f"Piper error: {result.stderr.decode()}")
                return b"", synth_time

            wav_data = Path(tmp_path).read_bytes()
            return wav_data, synth_time
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _start_playback(self, wav_data: bytes) -> subprocess.Popen | None:
        """Start non-blocking WAV playback from memory via aplay stdin."""
        if not wav_data:
            return None
        proc = subprocess.Popen(
            ["aplay", "-D", self.aplay_device],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        proc.stdin.write(wav_data)
        proc.stdin.close()
        return proc

    def _wait_for_pre(self, pre_proc):
        """Wait for a pre-play sound — handles both Popen and _DelayedSound."""
        if isinstance(pre_proc, _DelayedSound):
            pre_proc.wait()
        else:
            self._wait_for_playback(pre_proc)

    def _wait_for_playback(self, proc: subprocess.Popen | None, poll_interval: float = 0.05):
        """Wait for playback process to finish."""
        if proc is None:
            return

        try:
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode().strip() if proc.stderr else ""
                if stderr:
                    log.error(f"Playback error: {stderr}")
        finally:
            if proc.stderr:
                proc.stderr.close()

    def speak(self, text: str):
        """Synthesize and play a complete text (blocking)."""
        if not text.strip():
            return
        wav_data, synth_time = self.synthesize(text)
        log.info(f"TTS: {synth_time:.2f}s synth")
        proc = self._start_playback(wav_data)
        self._wait_for_playback(proc)

    def stream_speak(
        self,
        token_stream: Generator[str, None, None],
        latency=None,
        pre_proc: subprocess.Popen | None = None,
    ) -> Generator[str, None, None]:
        """Consume a token stream, buffer sentences, and speak each one as soon as ready.

        Playback is non-blocking: while sentence N plays via aplay, we continue
        consuming tokens and synthesizing sentence N+1, eliminating gaps.

        Yields each token for display/logging purposes.

        Args:
            token_stream: generator yielding text tokens from the LLM.
            latency: optional LatencyRecord to track TTS first-chunk timing.
            pre_proc: optional Popen for a thinking sound playing in parallel.
                      Waited on before the first TTS sentence plays to prevent overlap.

        Usage:
            for text in tts.stream_speak(llm_token_generator):
                print(text, end="", flush=True)
        """
        buffer = ""
        play_proc: subprocess.Popen | None = None
        stream_start = time.perf_counter()
        first_chunk_recorded = False

        for token in token_stream:
            buffer += token
            yield token  # Pass through for display

            # Check for sentence boundary
            sentences = SENTENCE_RE.split(buffer)
            if len(sentences) > 1:
                match = list(SENTENCE_RE.finditer(buffer))
                if match:
                    last_match = match[-1]
                    complete = buffer[: last_match.end()].strip()
                    buffer = buffer[last_match.end() :]

                    if complete:
                        log.debug(f"TTS sentence: '{complete}'")
                        wav_data, synth_time = self.synthesize(complete)

                        # Before first TTS playback, wait for thinking sound to finish
                        if pre_proc is not None:
                            self._wait_for_pre(pre_proc)
                            pre_proc = None

                        # Wait for previous playback, then start new one (non-blocking)
                        self._wait_for_playback(play_proc)
                        play_proc = self._start_playback(wav_data)

                        # Track time to first audio chunk
                        if not first_chunk_recorded and latency is not None:
                            latency.tts_first_chunk_ms = (time.perf_counter() - stream_start) * 1000
                            first_chunk_recorded = True

        # Speak any remaining text in buffer
        remaining = buffer.strip()
        if remaining:
            wav_data, synth_time = self.synthesize(remaining)
            log.debug(f"TTS remainder {synth_time:.2f}s: '{remaining[:60]}'")

            if pre_proc is not None:
                self._wait_for_pre(pre_proc)
                pre_proc = None
            self._wait_for_playback(play_proc)
            play_proc = self._start_playback(wav_data)

        # If no text was produced at all (e.g. model went straight to a tool call),
        # still wait for the thinking sound so it doesn't overlap with the next
        # round's TTS, which shares the same exclusive audio device.
        if pre_proc is not None:
            self._wait_for_pre(pre_proc)

        # Wait for final playback to complete
        self._wait_for_playback(play_proc)
