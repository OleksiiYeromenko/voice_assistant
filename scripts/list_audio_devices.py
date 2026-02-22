#!/usr/bin/env python3
"""List audio devices — find the right mic_device_index for config.yaml.

Usage: uv run scripts/list_audio_devices.py
"""

import ctypes

# Silence ALSA warnings
_EH = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def _h(*_): pass
_c = _EH(_h)
try:
    _a = ctypes.cdll.LoadLibrary("libasound.so")
    _a.snd_lib_error_set_handler(_c)
except OSError:
    pass

import pyaudio

pa = pyaudio.PyAudio()

print("=" * 60)
print("AUDIO DEVICES")
print("=" * 60)

input_devices = []
output_devices = []

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    name = info["name"]
    ins = info["maxInputChannels"]
    outs = info["maxOutputChannels"]
    rate = int(info["defaultSampleRate"])

    if ins > 0:
        input_devices.append((i, name, ins, rate))
    if outs > 0:
        output_devices.append((i, name, outs, rate))

print("\n🎤 INPUT (microphone) devices:")
for idx, name, ch, rate in input_devices:
    print(f"  index={idx}  channels={ch}  rate={rate}  name='{name}'")

print("\n🔊 OUTPUT (speaker) devices:")
for idx, name, ch, rate in output_devices:
    print(f"  index={idx}  channels={ch}  rate={rate}  name='{name}'")

# Suggest the USB mic
print("\n" + "=" * 60)
usb_inputs = [(i, n, c, r) for i, n, c, r in input_devices if "usb" in n.lower()]
if usb_inputs:
    idx = usb_inputs[0][0]
    print(f"✅ USB mic found: index={idx}, rate={usb_inputs[0][3]} Hz")
    print(f"   Device: '{usb_inputs[0][1]}'")
    print()
    print("   The assistant auto-detects USB mics when mic_device_index is null.")
    print(f"   To force this device: set mic_device_index: {idx} in config/config.yaml")
else:
    print("⚠  No USB mic found. Check 'arecord -l' output.")
    default = pa.get_default_input_device_info()
    print(f"   Default input: index={default['index']} '{default['name']}'")

pa.terminate()