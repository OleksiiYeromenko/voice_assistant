"""Streaming TTS using Piper — synthesize per-sentence, play immediately.

Key idea: as the LLM streams tokens, we buffer until we hit a sentence boundary
(sentence-ending punctuation), then synthesize that sentence with Piper and play it via aplay.
This means the user hears audio while the LLM is still generating.
"""

import logging
import re
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
        OUTPUT_DIR.mkdir(exist_ok=True)

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

    def _wait_for_playback(self, proc: subprocess.Popen | None):
        """Wait for a non-blocking playback process to finish and log errors."""
        if proc is None:
            return
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode().strip() if proc.stderr else ""
            if stderr:
                log.error(f"Playback error: {stderr}")

    def speak(self, text: str):
        """Synthesize and play a complete text."""
        if not text.strip():
            return
        path, synth_time, duration = self.synthesize(text)
        log.info(f"TTS: {synth_time:.2f}s synth, {duration:.2f}s audio")
        self.play(path)

    def stream_speak(self, token_stream: Generator[str, None, None]) -> Generator[str, None, None]:
        """Consume a token stream, buffer sentences, and speak each one as soon as ready.

        Playback is non-blocking: while sentence N plays via aplay, we continue
        consuming tokens and synthesizing sentence N+1, eliminating gaps.

        Yields each token for display/logging purposes.

        Usage:
            for text in tts.stream_speak(llm_token_generator):
                print(text, end="", flush=True)
        """
        buffer = ""
        play_proc: subprocess.Popen | None = None

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
                    buffer = buffer[last_match.end():]

                    if complete:
                        log.debug(f"TTS sentence: '{complete}'")
                        path, synth_time, duration = self.synthesize(complete)

                        # Wait for previous playback, then start new one (non-blocking)
                        self._wait_for_playback(play_proc)
                        play_proc = subprocess.Popen(
                            ["aplay", "-D", self.aplay_device, path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )

        # Speak any remaining text in buffer
        remaining = buffer.strip()
        if remaining:
            log.debug(f"TTS remainder: '{remaining}'")
            path, _, _ = self.synthesize(remaining)
            self._wait_for_playback(play_proc)
            play_proc = subprocess.Popen(
                ["aplay", "-D", self.aplay_device, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        # Wait for final playback to complete before returning
        self._wait_for_playback(play_proc)