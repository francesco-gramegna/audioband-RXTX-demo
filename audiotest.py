import numpy as np
import sounddevice as sd
def callback(indata, frames, time, status):
    print(np.max(indata))


print("\n==== Available Devices ====")
print(sd.query_devices())

info = sd.query_devices(3)
print(info)



pipewire_output = None
for idx, dev in enumerate(sd.query_devices()):
    if "pipewire" in dev['name'].lower() or "pulse" in dev['name'].lower():
        pipewire_output = idx
        break

sd.default.device = pipewire_output

with sd.InputStream(device=None , callback=callback, channels=1, samplerate=8000):

    input("Speak and press Enter...\n")



