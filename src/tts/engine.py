"""Streaming TTS using Piper — synthesize per-sentence, play immediately.

Key idea: as the LLM streams tokens, we buffer until we hit a sentence boundary
(sentence-ending punctuation), then synthesize that sentence with Piper and play it via aplay.
This means the user hears audio while the LLM is still generating.
"""

import logging
import math
import re
import struct
import subprocess
import time
import wave
from collections.abc import Generator
from pathlib import Path

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("./tts_output")
SENTENCE_RE = re.compile(r'\.(?!\d)\s+|[!?;]\s+')


class TTSEngine:
    def __init__(
        self,
        voice: str = "./voices/en_US-hfc_male-medium.onnx",
        aplay_device: str = "plughw:0,0",
    ):
        self.voice = voice
        self.aplay_device = aplay_device
        self._interrupted = False
        self._beep_path: str | None = None
        OUTPUT_DIR.mkdir(exist_ok=True)

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
        """Generate a short beep WAV if it doesn't exist."""
        path = OUTPUT_DIR / "beep.wav"
        if path.exists():
            self._beep_path = str(path)
            return

        sample_rate = 22050
        duration = 0.15  # 150 ms
        freq = 880  # A5 note
        n_samples = int(sample_rate * duration)

        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(n_samples):
                t = i / sample_rate
                # Fade in/out to avoid clicks
                envelope = min(t / 0.01, 1.0) * min((duration - t) / 0.01, 1.0)
                sample = int(16000 * envelope * math.sin(2 * math.pi * freq * t))
                wf.writeframes(struct.pack("<h", max(-32768, min(32767, sample))))

        self._beep_path = str(path)
        log.debug("Beep WAV generated")

    def play_beep(self) -> subprocess.Popen | None:
        """Play acknowledgment beep (non-blocking). Returns Popen or None."""
        self._ensure_beep()
        try:
            return subprocess.Popen(
                ["aplay", "-D", self.aplay_device, self._beep_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.warning(f"Beep failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Core TTS
    # ------------------------------------------------------------------
    def synthesize(self, text: str) -> tuple[str, float, float]:
        """Synthesize text to WAV file. Returns (path, synth_time_s, audio_duration_s)."""
        output_path = str(OUTPUT_DIR / f"tts_{int(time.time() * 1000)}.wav")

        start = time.perf_counter()
        result = subprocess.run(
            ["piper", "--model", self.voice, "--output_file", output_path],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        synth_time = time.perf_counter() - start

        if result.returncode != 0:
            log.error(f"Piper error: {result.stderr.decode()}")
            return output_path, synth_time, 0.0

        try:
            with wave.open(output_path, "rb") as wf:
                audio_duration = wf.getnframes() / wf.getframerate()
        except Exception:
            audio_duration = 0.0

        return output_path, synth_time, audio_duration

    def play(self, path: str):
        """Play WAV via aplay (blocking)."""
        result = subprocess.run(
            ["aplay", "-D", self.aplay_device, path],
            capture_output=True,
        )
        if result.returncode != 0:
            log.error(f"Playback error: {result.stderr.decode().strip()}")

    def _wait_for_playback(self, proc: subprocess.Popen | None, poll_interval: float = 0.05):
        """Wait for playback process, polling periodically for interrupt.

        Returns True if playback completed normally, False if interrupted.
        """
        if proc is None:
            return True

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

    def speak(self, text: str):
        """Synthesize and play a complete text."""
        if not text.strip():
            return
        path, synth_time, duration = self.synthesize(text)
        log.info(f"TTS: {synth_time:.2f}s synth, {duration:.2f}s audio")
        self.play(path)

    def stream_speak(
        self,
        token_stream: Generator[str, None, None],
        latency=None,
    ) -> Generator[str, None, None]:
        """Consume a token stream, buffer sentences, and speak each one as soon as ready.

        Playback is non-blocking: while sentence N plays via aplay, we continue
        consuming tokens and synthesizing sentence N+1, eliminating gaps.

        Yields each token for display/logging purposes.

        Args:
            token_stream: generator yielding text tokens from the LLM.
            latency: optional LatencyRecord to track TTS first-chunk timing.

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
                        path, synth_time, duration = self.synthesize(complete)

                        # Wait for previous playback, then start new one (non-blocking)
                        if not self._wait_for_playback(play_proc):
                            break  # Interrupted during wait
                        play_proc = subprocess.Popen(
                            ["aplay", "-D", self.aplay_device, path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )

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
                path, _, _ = self.synthesize(remaining)
                if self._wait_for_playback(play_proc):
                    play_proc = subprocess.Popen(
                        ["aplay", "-D", self.aplay_device, path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )

        # Wait for final playback to complete (or interrupted)
        self._wait_for_playback(play_proc)