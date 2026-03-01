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
import time
import wave
from collections.abc import Generator
from pathlib import Path

log = logging.getLogger(__name__)

SENTENCE_RE = re.compile(r'\.(?!\d)\s+|[!?;]\s+')


class TTSEngine:
    def __init__(
        self,
        voice: str = "./voices/en_US-hfc_male-medium.onnx",
        aplay_device: str = "plughw:0,0",
        sounds_dir: str = "./sounds",
    ):
        self.voice = voice
        self.aplay_device = aplay_device
        self._interrupted = False
        self._beep_wav: bytes | None = None
        self._startup_sounds: list[Path] = []
        self._greeting_sounds: list[Path] = []
        self._thinking_sounds: list[Path] = []
        self._load_sounds(sounds_dir)

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

    # ------------------------------------------------------------------
    # Interrupt mechanism
    # ------------------------------------------------------------------
    def interrupt(self):
        """Signal TTS to stop playback immediately."""
        self._interrupted = True

    def _reset_interrupt(self):
        self._interrupted = False

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

    def play_thinking(self) -> subprocess.Popen | None:
        """Play a random thinking/processing sound (non-blocking). Returns None if none found."""
        if not self._thinking_sounds:
            return None
        return self._start_playback(random.choice(self._thinking_sounds).read_bytes())

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

    def _wait_for_playback(self, proc: subprocess.Popen | None, poll_interval: float = 0.05):
        """Wait for playback process, polling periodically for interrupt.

        Returns True if playback completed normally, False if interrupted.
        """
        if proc is None:
            return True

        try:
            while proc.poll() is None:
                if self._interrupted:
                    proc.terminate()
                    proc.wait()
                    return False
                time.sleep(poll_interval)

            if proc.returncode != 0:
                stderr = proc.stderr.read().decode().strip() if proc.stderr else ""
                if stderr:
                    log.error(f"Playback error: {stderr}")
            return True
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
        self._reset_interrupt()
        buffer = ""
        play_proc: subprocess.Popen | None = None
        stream_start = time.perf_counter()
        first_chunk_recorded = False

        for token in token_stream:
            if self._interrupted:
                break

            buffer += token
            yield token  # Pass through for display

            # Check for sentence boundary
            sentences = SENTENCE_RE.split(buffer)
            if len(sentences) > 1:
                match = list(SENTENCE_RE.finditer(buffer))
                if match:
                    last_match = match[-1]
                    complete = buffer[: last_match.end()].strip()
                    buffer = buffer[last_match.end():]

                    if complete:
                        log.debug(f"TTS sentence: '{complete}'")
                        wav_data, synth_time = self.synthesize(complete)

                        if self._interrupted:
                            break  # Interrupted during synthesis — discard result

                        # Before first TTS playback, wait for thinking sound to finish
                        if pre_proc is not None:
                            if not self._wait_for_playback(pre_proc):
                                break  # Interrupted during thinking sound
                            pre_proc = None

                        # Wait for previous playback, then start new one (non-blocking)
                        if not self._wait_for_playback(play_proc):
                            break  # Interrupted during wait
                        play_proc = self._start_playback(wav_data)

                        # Track time to first audio chunk
                        if not first_chunk_recorded and latency is not None:
                            latency.tts_first_chunk_ms = (
                                time.perf_counter() - stream_start
                            ) * 1000
                            first_chunk_recorded = True

        # Speak any remaining text in buffer (skip if interrupted)
        if not self._interrupted:
            remaining = buffer.strip()
            if remaining:
                log.debug(f"TTS remainder: '{remaining}'")
                wav_data, _ = self.synthesize(remaining)
                # In case no sentence boundary was hit, still wait for thinking sound
                if pre_proc is not None:
                    self._wait_for_playback(pre_proc)
                    pre_proc = None
                if self._wait_for_playback(play_proc):
                    play_proc = self._start_playback(wav_data)

        # Wait for final playback to complete (or interrupted)
        self._wait_for_playback(play_proc)
