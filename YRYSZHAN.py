import os
import sys
import time
import json
import cv2
import torch
import pyttsx3
import pyaudio
import re
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from vosk import Model as VoskModel, KaldiRecognizer

IMAGE_PATH = 'input.jpg'
WEIGHTS = 'v5lite-s.pt'     
IMG_SIZE = 320
DEVICE = 'cpu'             
CONF_THRES = 0.25
IOU_THRES  = 0.45

VOSK_MODEL_DIR = "models/vosk-model-small-ru-0.22"
SAMPLE_RATE = 16000
CHUNK_SIZE  = 8192
CHANNELS    = 1
MIC_INDEX   = None  

WAKE_PHRASES = ("вперед", "вперёд", "что впереди есть", "что впереди")
SILENCE_AFTER_SPEAK_SEC = 0.3

VOLUME_INIT = 0.8
TTS_MUTED = False
CONTINUOUS_MODE = False
CONTINUOUS_PERIOD = 3.0  
CONF_THRES_VAR = CONF_THRES 

TRANSLATE = {
    "person":"человек","bicycle":"велосипед","car":"машина","motorcycle":"мотоцикл","airplane":"самолёт",
    "bus":"автобус","train":"поезд","truck":"грузовик","boat":"лодка","traffic light":"светофор",
    "fire hydrant":"пожарный гидрант","stop sign":"знак стоп","parking meter":"парковочный счётчик",
    "bench":"скамейка","bird":"птица","cat":"кот","dog":"собака","horse":"лошадь","sheep":"овца","cow":"корова",
    "elephant":"слон","bear":"медведь","zebra":"зебра","giraffe":"жираф","backpack":"рюкзак","umbrella":"зонт",
    "handbag":"сумка","tie":"галстук","frisbee":"фрисби","skis":"лыжи","snowboard":"сноуборд",
    "sports ball":"мяч","tennis racket":"теннисная ракетка","bottle":"бутылка","wine glass":"бокал",
    "cup":"чашка","fork":"вилка","knife":"нож","spoon":"ложка","bowl":"миска","banana":"банан","apple":"яблоко",
    "sandwich":"бутерброд","orange":"апельсин","broccoli":"брокколи","carrot":"морковь","hot dog":"хот-дог",
    "pizza":"пицца","donut":"пончик","cake":"торт","chair":"стул","couch":"диван","potted plant":"растение",
    "bed":"кровать","dining table":"обеденный стол","toilet":"туалет","tv":"телевизор","laptop":"ноутбук",
    "mouse":"мышь","remote":"пульт","keyboard":"клавиатура","cell phone":"телефон","microwave":"микроволновка",
    "oven":"духовка","toaster":"тостер","refrigerator":"холодильник","book":"книга","clock":"часы","vase":"ваза",
    "scissors":"ножницы","teddy bear":"плюшевый мишка","hair drier":"фен","toothbrush":"зубная щётка"
}

def init_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('voice', 'ru')   
    engine.setProperty('rate', 150)
    engine.setProperty('volume', VOLUME_INIT)  # 0.0..1.0
    return engine

def speak(engine, text):
    global TTS_MUTED
    if TTS_MUTED:
        print("[MUTED]", text)
        return
    print("[SAY] " + text)
    engine.say(text)
    engine.runAndWait()
    time.sleep(SILENCE_AFTER_SPEAK_SEC)

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAMERA] Не удалось открыть камеру")
        return None

    time.sleep(0.3)
    for _ in range(5):
        cap.read()
        time.sleep(0.03)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("[CAMERA] Кадр не получен")
        return None
    cv2.imwrite(IMAGE_PATH, frame)
    print(f"[CAMERA] Снимок сохранён: {IMAGE_PATH}")
    return IMAGE_PATH

def init_yolo():
    device = select_device(DEVICE)
    yolo_model = attempt_load(WEIGHTS, map_location=device)
    stride = int(yolo_model.stride.max())
    imgsz = check_img_size(IMG_SIZE, s=stride)
    yolo_model.eval()
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    return yolo_model, device, stride, imgsz, names

def detect_raw(image_path, yolo_model, device, stride, imgsz, names, conf_thres=None):
    if conf_thres is None:
        conf_thres = CONF_THRES_VAR
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)
    found = []
    for path, img, im0, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time.time()
        with torch.no_grad():
            pred = yolo_model(img)[0]
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=IOU_THRES)
        t2 = time.time()
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    found.append(label)
        print(f"[YOLO] Детекция заняла {(t2 - t1):.3f} сек.")
    return found

def run_detection(image_path, yolo_model, device, stride, imgsz, names):
    found = detect_raw(image_path, yolo_model, device, stride, imgsz, names)
    if not found:
        return "Впереди ничего не обнаружено."
    unique = sorted(set(found))
    translated = [TRANSLATE.get(o, o) for o in unique]
    return "Будьте внимательны! Впереди есть " + ", ".join(translated) + "."

def init_asr():
    if not os.path.isdir(VOSK_MODEL_DIR):
        print(f"[VOSK] Модель не найдена: {VOSK_MODEL_DIR}")
        sys.exit(1)
    vosk_model = VoskModel(VOSK_MODEL_DIR)
    asr = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=CHANNELS,
                     rate=SAMPLE_RATE,
                     input=True,
                     input_device_index=MIC_INDEX,
                     frames_per_buffer=CHUNK_SIZE)
    stream.start_stream()
    return pa, stream, asr

def phrase_triggers(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(p in t for p in WAKE_PHRASES)

def say_time(tts):
    now = datetime.now()
    speak(tts, f"Сейчас {now.strftime('%H:%M')}.")

def set_volume(tts, delta=None, absolute=None):
    v = tts.getProperty('volume')
    if absolute is not None:
        v = max(0.0, min(1.0, absolute))
    elif delta is not None:
        v = max(0.0, min(1.0, v + delta))
    tts.setProperty('volume', v)
    speak(tts, f"Громкость {int(v*100)} процентов.")

def count_people(found):
    return sum(1 for x in found if x == "person")

def handle_command(text, tts, yolo_model, device, stride, imgsz, names):

    global TTS_MUTED, CONTINUOUS_MODE, CONF_THRES_VAR

    t = (text or "").lower().strip()

    if any(w in t for w in ("стоп" , "тихо", "замолчи")):
        TTS_MUTED = True
        print("[TTS] muted")
        return True
    if any(w in t for w in ("говори", "включи голос", "озвучка")):
        TTS_MUTED = False
        speak(tts, "Голос включён.")
        return True

    if "громче" in t:
        set_volume(tts, delta=+0.1); return True
    if "тише" in t:
        set_volume(tts, delta=-0.1); return True
    if "громкость" in t:
        m = re.search(r"громк(?:ость)?\s+(\d{1,3})", t)
        if m:
            pct = int(m.group(1))
            set_volume(tts, absolute=max(0, min(100, pct))/100.0)
            return True

    if "скажи время" in t or "который час" in t:
        say_time(tts); return True

    if "сделай фото" in t or "сфотографируй" in t:
        path = capture_image()
        if path:
            speak(tts, "Фото сохранено.")
        return True

    if "режим поток" in t or "авто режим" in t or "потоковый режим" in t:
        if any(w in t for w in ("выключи", "отключи", "стоп")):
            CONTINUOUS_MODE = False
            speak(tts, "Потоковый режим выключен.")
        else:
            CONTINUOUS_MODE = True
            speak(tts, f"Потоковый режим включён. Интервал {int(CONTINUOUS_PERIOD)} секунд.")
        return True

    if "сколько людей" in t:
        path = capture_image()
        if path:
            found = detect_raw(path, yolo_model, device, stride, imgsz, names)
            n = count_people(found)
            speak(tts, f"Я вижу {n} " + ("человека." if 1 <= n <= 4 else "человек."))
        return True

    if "выход" in t or "заверши" in t or "закрыть" in t:
        speak(tts, "Выход.")
        return "EXIT"

    return False

def main():
    print("[INIT] Загружаю YOLOv5-Lite...")
    yolo_model, device, stride, imgsz, names = init_yolo()
    print("[INIT] YOLO готов.")

    print("[INIT] Готовлю распознавание речи (Vosk)...")
    pa, stream, asr = init_asr()
    print("[INIT] ASR готов. Скажи: 'вперед' или 'что впереди есть'.")

    tts = init_tts_engine()
    last_auto = 0.0

    try:
        while True:
            if CONTINUOUS_MODE and (time.time() - last_auto) >= CONTINUOUS_PERIOD:
                path = capture_image()
                if path:
                    msg = run_detection(path, yolo_model, device, stride, imgsz, names)
                    speak(tts, msg)
                last_auto = time.time()

            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            if asr.AcceptWaveform(data):
                r = json.loads(asr.Result())
                text = r.get("text", "")
                if text:
                    print("\r[YOU] " + text)

                    res = handle_command(text, tts, yolo_model, device, stride, imgsz, names)
                    if res == "EXIT":
                        break
                    if res is True:
                        continue  

                    if phrase_triggers(text):
                        path = capture_image()
                        if path:
                            msg = run_detection(path, yolo_model, device, stride, imgsz, names)
                            speak(tts, msg)
            else:
                pr = json.loads(asr.PartialResult())
                partial = pr.get("partial", "")
                if partial:
                    sys.stdout.write("\r[...]" + partial + "   ")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[EXIT] Остановка пользователем.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
