import pyaudio
from vosk import Model, KaldiRecognizer

MODEL_PATH = "models/vosk-model-small-ru-0.22/"

pa = pyaudio.PyAudio()

# найдём индекс устройства по подстроке имени
target_substr = "USB Audio Device".lower()
input_index = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if int(info.get("maxInputChannels", 0)) > 0 and target_substr in info.get("name", "").lower():
        input_index = i
        default_rate = int(info.get("defaultSampleRate", 16000))
        break

if input_index is None:
    raise RuntimeError("Не нашёл устройство с именем, содержащим 'USB Audio Device'.")

print(f"Использую устройство [{input_index}]: {info['name']} @ {default_rate} Гц")

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, default_rate)

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=default_rate,            # можно 16000, если устройство поддерживает
    input=True,
    input_device_index=input_index,
    frames_per_buffer=8192,
)

print("Говори... (Ctrl+C для выхода)")
try:
    while True:
        data = stream.read(8192, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            print(rec.Result())
        else:
            # print(rec.PartialResult())  # если нужно показывать partial
            pass
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
