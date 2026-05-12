# Implement Custom Wake Word Verifier

## Goal

Add a second-stage verifier model to the wake word detector to reduce false positives while
enabling a lower main threshold for better recall. The verifier is trained with openWakeWord's
built-in `train_custom_verifier` on labeled positive/negative audio captures, then loaded at
runtime to confirm or reject detections before they fire the assistant.

**No new dependencies.** `scikit-learn`, `scipy`, and `numpy` are already in `pyproject.toml`.
The verifier is a plain sklearn `LogisticRegression` pkl using AudioSet embeddings already
computed by the openWakeWord model on every frame.

---

## How it works

Every 80ms audio chunk goes through this pipeline:

```
arecord → 1280 samples
     ↓
oww.predict(chunk)  →  raw_score (0.0–1.0)
     ↓  only if raw_score >= verifier_threshold (e.g. 0.1)
oww.preprocessor.get_features(model_inputs[stem])  →  embedding window
verifier.predict_proba(embedding)  →  verifier_score (0.0–1.0)
     ↓  effective_score = verifier_score  (or raw_score if verifier disabled)
     ↓  if effective_score >= threshold (e.g. 0.70)
save ring buffer → "20260511_192226_0.714_0.923.wav"  (raw_score + verifier_score)
oww.reset() → yield → state machine → STT → LLM → TTS
```

- Verifier runs only on score spikes, not every frame — negligible latency (~0.5ms).
- The verifier **replaces** the OWW score; existing threshold logic and ring-buffer save are unchanged.
- Score in WAV filename: `{ts}_{raw:.3f}_{verifier:.3f}.wav` when enabled, `{ts}_{raw:.3f}.wav` when disabled — backward compatible.
- Toggle on/off: set `verifier_model: null` in config.yaml.

---

## Existing code reference

### `src/wake_word/detector.py` — current `__init__` (lines 46–73)

```python
class WakeWordDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.7,
        alsa_device: str | None = None,
        record_detections: bool = False,
    ):
        import openwakeword
        from openwakeword.model import Model

        if not model.endswith(".onnx"):
            openwakeword.utils.download_models()

        self.model_name = model
        self.threshold = threshold
        self._alsa_device = alsa_device
        self.oww = Model(wakeword_models=[model])

        self._record_detections = record_detections
        if record_detections:
            self._capture_dir = Path("data/wake_captures")
            for sub in ("raw", "false_positives", "true_positives"):
                (self._capture_dir / sub).mkdir(parents=True, exist_ok=True)
            log.info(f"Wake capture recording enabled → {self._capture_dir}")

        log.info(f"Wake word detector ready: '{model}' (threshold={threshold})")
```

### `src/wake_word/detector.py` — current `_save_capture` (lines 75–83)

```python
    def _save_capture(self, ring: deque, score: float) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._capture_dir / "raw" / f"{ts}_{score:.3f}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(ring))
        log.debug(f"Saved wake capture: {path.name}")
```

### `src/wake_word/detector.py` — current detection block in `detect_once` (lines 139–146)

```python
                audio_16k = np.frombuffer(data, dtype=np.int16)
                prediction = self.oww.predict(audio_16k)

                for model_name, score in prediction.items():
                    if score >= self.threshold:
                        log.info(f"Wake word interrupt '{model_name}': {score:.3f}")
                        self.oww.reset()
                        return True
```

### `src/wake_word/detector.py` — current detection block in `listen` (lines 209–222)

```python
                        ring.append(data)
                        audio_16k = np.frombuffer(data, dtype=np.int16)
                        prediction = self.oww.predict(audio_16k)

                        for model_name, score in prediction.items():
                            if score >= self.threshold:
                                log.info(f"Wake word '{model_name}': {score:.3f}")
                                # Stop arecord BEFORE yielding so STT can use the device
                                _kill_proc(proc)
                                if self._record_detections:
                                    self._save_capture(ring, score)
                                self.oww.reset()
                                yield score
                                detected = True
                                break
```

### `src/main.py` — current `WakeWordDetector` instantiation (lines 548–553)

```python
            wake_detector = WakeWordDetector(
                model=ww_cfg["model"],
                threshold=ww_cfg["threshold"],
                alsa_device=stt_cfg.get("alsa_device"),
                record_detections=ww_cfg.get("record_detections", False),
            )
```

### `config/config.yaml` — current wake_word section

```yaml
wake_word:
  model: "./models/hey_Poon-dyk_acc_0.84_recall_0.69_fp_6.6.onnx"
  threshold: 0.99
  record_detections: true
```

---

## Changes to make

### 1. New file: `scripts/train_verifier.py`

Create this file:

```python
#!/usr/bin/env python3
"""Train a custom verifier model for the wake word detector.

Uses labeled captures from data/wake_captures/true_positives/ and
data/wake_captures/false_positives/ to train a LogisticRegression classifier
on openWakeWord's internal AudioSet embeddings.

Usage:
    uv run scripts/train_verifier.py
"""

from pathlib import Path

from openwakeword.custom_verifier_model import train_custom_verifier

MODEL = "./models/hey_Poon-dyk_acc_0.84_recall_0.69_fp_6.6.onnx"
TP_DIR = Path("data/wake_captures/true_positives")
FP_DIR = Path("data/wake_captures/false_positives")
OUT = "models/verifier.pkl"

tp = sorted(TP_DIR.glob("*.wav"))
fp = sorted(FP_DIR.glob("*.wav"))

if not tp:
    raise SystemExit(f"No positive examples found in {TP_DIR}")
if not fp:
    raise SystemExit(f"No negative examples found in {FP_DIR}")

print(f"Training verifier: {len(tp)} positive, {len(fp)} negative examples")
train_custom_verifier(tp, fp, output_path=OUT, model_name=MODEL)
print(f"Saved → {OUT}")
```

### 2. Modify `src/wake_word/detector.py`

**Change `__init__`** — replace the existing signature and body with:

```python
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.7,
        alsa_device: str | None = None,
        record_detections: bool = False,
        verifier_model: str | None = None,
        verifier_threshold: float = 0.1,
    ):
        import pickle

        import openwakeword
        from openwakeword.model import Model

        if not model.endswith(".onnx"):
            openwakeword.utils.download_models()

        self.model_name = model
        self.threshold = threshold
        self._alsa_device = alsa_device
        self.oww = Model(wakeword_models=[model])
        self._model_stem = Path(model).stem
        self._verifier = None
        self._verifier_threshold = verifier_threshold

        if verifier_model:
            with open(verifier_model, "rb") as f:
                self._verifier = pickle.load(f)
            log.info(f"Verifier loaded: {verifier_model} (gate={verifier_threshold})")

        self._record_detections = record_detections
        if record_detections:
            self._capture_dir = Path("data/wake_captures")
            for sub in ("raw", "false_positives", "true_positives"):
                (self._capture_dir / sub).mkdir(parents=True, exist_ok=True)
            log.info(f"Wake capture recording enabled → {self._capture_dir}")

        log.info(f"Wake word detector ready: '{model}' (threshold={threshold})")
```

**Change `_save_capture`** — replace existing method:

```python
    def _save_capture(self, ring: deque, raw_score: float, verifier_score: float | None = None) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_part = f"{raw_score:.3f}_{verifier_score:.3f}" if verifier_score is not None else f"{raw_score:.3f}"
        path = self._capture_dir / "raw" / f"{ts}_{score_part}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(ring))
        log.debug(f"Saved wake capture: {path.name}")
```

**Change detection block in `detect_once`** — replace the existing inner loop body (lines 139–146):

```python
                audio_16k = np.frombuffer(data, dtype=np.int16)
                prediction = self.oww.predict(audio_16k)

                for model_name, raw_score in prediction.items():
                    verifier_score = None
                    effective_score = raw_score
                    if self._verifier and raw_score >= self._verifier_threshold:
                        features = self.oww.preprocessor.get_features(
                            self.oww.model_inputs[self._model_stem]
                        )
                        verifier_score = float(self._verifier.predict_proba(features)[0][-1])
                        effective_score = verifier_score
                    if effective_score >= self.threshold:
                        log.info(
                            f"Wake word interrupt '{model_name}': "
                            f"raw={raw_score:.3f} verifier={verifier_score}"
                        )
                        self.oww.reset()
                        return True
```

**Change detection block in `listen`** — replace the existing inner loop body (lines 209–222):

```python
                        ring.append(data)
                        audio_16k = np.frombuffer(data, dtype=np.int16)
                        prediction = self.oww.predict(audio_16k)

                        for model_name, raw_score in prediction.items():
                            verifier_score = None
                            effective_score = raw_score
                            if self._verifier and raw_score >= self._verifier_threshold:
                                features = self.oww.preprocessor.get_features(
                                    self.oww.model_inputs[self._model_stem]
                                )
                                verifier_score = float(
                                    self._verifier.predict_proba(features)[0][-1]
                                )
                                effective_score = verifier_score
                            if effective_score >= self.threshold:
                                log.info(
                                    f"Wake word '{model_name}': "
                                    f"raw={raw_score:.3f} verifier={verifier_score}"
                                )
                                _kill_proc(proc)
                                if self._record_detections:
                                    self._save_capture(ring, raw_score, verifier_score)
                                self.oww.reset()
                                yield effective_score
                                detected = True
                                break
```

### 3. Modify `config/config.yaml`

Replace the `wake_word` section:

```yaml
wake_word:
  model: "./models/hey_Poon-dyk_acc_0.84_recall_0.69_fp_6.6.onnx"
  threshold: 0.70
  record_detections: true
  verifier_model: "./models/verifier.pkl"
  verifier_threshold: 0.1
```

### 4. Modify `src/main.py`

Replace the `WakeWordDetector(...)` call (lines 548–553):

```python
            wake_detector = WakeWordDetector(
                model=ww_cfg["model"],
                threshold=ww_cfg["threshold"],
                alsa_device=stt_cfg.get("alsa_device"),
                record_detections=ww_cfg.get("record_detections", False),
                verifier_model=ww_cfg.get("verifier_model"),
                verifier_threshold=ww_cfg.get("verifier_threshold", 0.1),
            )
```

---

## Verification

1. **Train the model first:**
   ```bash
   uv run scripts/train_verifier.py
   # Expect: "X positive, Y negative examples" then "Saved → models/verifier.pkl"
   ```

2. **Check no regressions in non-wake modes:**
   ```bash
   uv run python -m src.main --no-wake
   uv run python -m src.main --text
   ```

3. **Test wake word mode:**
   ```bash
   uv run python -m src.main
   ```
   - Speak wake word 10 times: expect 8+ detections (better than old 0.99 threshold recall)
   - Play background speech/TV for 2+ minutes: confirm FP rate stays low
   - Check `data/wake_captures/raw/` — filenames should show dual scores, e.g. `20260511_192226_0.714_0.923.wav`

4. **Test the off switch:**
   - Set `verifier_model: null` in config.yaml
   - Restart assistant — filenames should revert to single score

5. **Check logs** for lines like:
   ```
   Wake word 'hey_Poon-dyk...': raw=0.714 verifier=0.923
   ```
