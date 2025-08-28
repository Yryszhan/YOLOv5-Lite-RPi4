import os
import sys
import time
import json
import cv2
import torch
import pyttsx3
import pyaudio
import re
import serial
import requests
import pigpio
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from vosk import Model as VoskModel, KaldiRecognizer

# ===================== НАСТРОЙКИ =====================
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
MIC_INDEX   = None  # None = микрофон по умолчанию

# Голосовые фразы
WAKE_PHRASES = ("вперед", "вперёд", "что впереди есть", "что впереди")
SILENCE_AFTER_SPEAK_SEC = 0.3

# Речь
VOLUME_INIT = 0.8
TTS_MUTED = False

# Режим автодетекции
CONTINUOUS_MODE = False
CONTINUOUS_PERIOD = 3.0
CONF_THRES_VAR = CONF_THRES

# Telegram — из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# GPS (через NEO-6 на GPIO UART)
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
GPS_READ_TIMEOUT = 8.0  # сек ждать фикса

# ===== HC-SR04 (РАДАР) =====
# Пины BCM: TRIG -> GPIO23, ECHO -> GPIO24 (ECHO через делитель до 3.3В!)
SONAR_TRIG = 23
SONAR_ECHO = 24
RADAR_ENABLED = False
RADAR_PERIOD = 0.5         # как часто опрашивать сонар, сек
RADAR_SAY_COOLDOWN = 3.0   # не говорить слишком часто, сек
RADAR_THRESHOLD_CM = 40.0  # порог (говорить, если ближе)
RADAR_DEBUG = True         # печатать измеренную дистанцию/таймауты
TEMP_C = 20.0              # для скорости звука
SOUND_SPEED = 331.3 + 0.606 * TEMP_C  # м/с
GLITCH_US = 150            # подавление дребезга ECHO (микросекунды)

PGPIO = None               # инстанс pigpio.pi()

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

# ===================== РЕЧЬ =====================
def init_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('voice', 'ru')   # выбери русскую голосовую
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

# ===================== КАМЕРА / YOLO =====================
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

# ===================== VOSK (ASR) =====================
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

# ===================== GPS + Telegram =====================
def _dm_to_deg(dm: str, dir_letter: str):
    if not dm:
        return None
    try:
        val = float(dm)
    except Exception:
        return None
    deg = int(val // 100)
    minutes = val - deg * 100
    dec = deg + minutes / 60.0
    if dir_letter in ("S", "W", "s", "w"):
        dec = -dec
    return dec

def get_current_gps(timeout_s: float = GPS_READ_TIMEOUT):
    end = time.time() + max(1.0, timeout_s)
    last_candidate = None
    try:
        with serial.Serial(GPS_PORT, GPS_BAUD, timeout=1) as ser:
            while time.time() < end:
                line = ser.readline().decode(errors="ignore").strip()
                if not line.startswith("$"):
                    continue
                if line.startswith(("$GPRMC", "$GNRMC")):
                    parts = line.split(",")
                    if len(parts) >= 7:
                        status = parts[2]
                        lat = _dm_to_deg(parts[3], parts[4] if len(parts) > 4 else "")
                        lon = _dm_to_deg(parts[5], parts[6] if len(parts) > 6 else "")
                        if lat is not None and lon is not None:
                            last_candidate = (lat, lon)
                            if status == "A":
                                return (lat, lon)
                elif line.startswith(("$GPGGA", "$GNGGA")):
                    parts = line.split(",")
                    if len(parts) >= 7:
                        lat = _dm_to_deg(parts[2], parts[3] if len(parts) > 3 else "")
                        lon = _dm_to_deg(parts[4], parts[5] if len(parts) > 5 else "")
                        try:
                            qual = int(parts[6] or "0")
                        except Exception:
                            qual = 0
                        if lat is not None and lon is not None:
                            last_candidate = (lat, lon)
                            if qual >= 1:
                                return (lat, lon)
    except Exception as e:
        print(f"[GPS] Ошибка доступа к {GPS_PORT}: {e}")
        return None
    return last_candidate

def send_telegram_help(lat: float=None, lon: float=None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Заполни TELEGRAM_TOKEN и TELEGRAM_CHAT_ID!")
        return False
    text = "Мне нужна помощь!"
    if lat is not None and lon is not None:
        link = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
        text += f"\nКоординаты: {lat:.6f}, {lon:.6f}\n{link}"
    else:
        text += "\nКоординаты не получены."
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True
        }, timeout=15)
        ok = False
        desc = r.text[:160]
        try:
            jr = r.json()
            ok = jr.get("ok", False)
            desc = jr.get("description", desc)
        except Exception:
            pass
        print(f"[TG] status={r.status_code} ok={ok} desc={desc}")
        return ok
    except requests.exceptions.SSLError as e:
        print(f"[TG] SSL error: {e}  (проверь время системы и сертификаты)")
    except requests.exceptions.ConnectTimeout:
        print("[TG] Timeout: не удалось подключиться к api.telegram.org")
    except requests.exceptions.ConnectionError as e:
        print(f"[TG] ConnectionError: {e}  (интернет/прокси/DNS)")
    except Exception as e:
        print(f"[TG] Unexpected error: {e}")
    return False

# ===================== SONAR (HC-SR04) =====================
def init_sonar():
    """Инициализация pigpio и пинов. Возвращает True/False."""
    global PGPIO
    try:
        PGPIO = pigpio.pi()
    except Exception as e:
        print(f"[SONAR] pigpio.pi() ошибка: {e}")
        PGPIO = None
        return False
    if not PGPIO or not PGPIO.connected:
        print("[SONAR] pigpio не подключён. Запусти: sudo systemctl start pigpiod")
        PGPIO = None
        return False
    PGPIO.set_mode(SONAR_TRIG, pigpio.OUTPUT)
    PGPIO.set_mode(SONAR_ECHO, pigpio.INPUT)
    PGPIO.set_glitch_filter(SONAR_ECHO, GLITCH_US)  # подавление дребезга
    PGPIO.write(SONAR_TRIG, 0)
    time.sleep(0.05)
    print("[SONAR] Инициализировано.")
    return True

def sonar_distance_cm_once(timeout_s=0.06):
    """Одно измерение. Возвращает расстояние в см или None при таймауте."""
    if not PGPIO:
        return None
    # 10 мкс импульс на TRIG
    PGPIO.gpio_trigger(SONAR_TRIG, 10, 1)
    t0 = time.time()

    # Ждём фронт на ECHO
    while PGPIO.read(SONAR_ECHO) == 0:
        if time.time() - t0 > timeout_s:
            return None
    t1 = PGPIO.get_current_tick()

    # Ждём спад на ECHO
    while PGPIO.read(SONAR_ECHO) == 1:
        if time.time() - t0 > timeout_s:
            return None
    t2 = PGPIO.get_current_tick()

    us = pigpio.tickDiff(t1, t2)
    secs = us / 1e6
    dist_m = SOUND_SPEED * secs / 2.0
    return dist_m * 100.0  # см

def sonar_distance_cm(samples=3, pause=0.02):
    """Медиана нескольких измерений (устойчивее)."""
    vals = []
    for _ in range(samples):
        d = sonar_distance_cm_once()
        if d is not None:
            vals.append(d)
        time.sleep(pause)
    if not vals:
        return None
    vals.sort()
    mid = vals[len(vals)//2]
    return round(mid, 1)

# ===================== ОБРАБОТКА КОМАНД =====================
def handle_command(text, tts, yolo_model, device, stride, imgsz, names):
    global TTS_MUTED, CONTINUOUS_MODE, RADAR_ENABLED

    t = (text or "").lower().strip()

    # --- SOS команды ---
    if "мне нужна помощь" in t or re.search(r"\bпомощь\b", t):
        speak(tts, "Получаю координаты и отправляю сообщение в Телеграм.")
        coords = get_current_gps()
        if coords is None:
            speak(tts, "Не удалось получить координаты. Отправляю сообщение без ссылки.")
            ok = send_telegram_help(None, None)
        else:
            lat, lon = coords
            ok = send_telegram_help(lat, lon)
        if ok:
            speak(tts, "Готово. Сообщение отправлено.")
        else:
            speak(tts, "Ошибка отправки. Проверь интернет и настройки Telegram.")
        return True

    # --- РАДАР: включить/выключить ---
    if any(kw in t for kw in ("включить радар", "включи радар", "включить сонар", "включи сонар")):
        if PGPIO or init_sonar():
            RADAR_ENABLED = True
            speak(tts, "Радар включен.")
        else:
            RADAR_ENABLED = False
            speak(tts, "Радар недоступен. Запусти pigpio.")
        return True

    if any(kw in t for kw in ("выключить радар", "отключить радар", "выключи радар", "отключи радар",
                               "выключить сонар", "отключить сонар", "выключи сонар", "отключи сонар")):
        RADAR_ENABLED = False
        speak(tts, "Радар выключен.")
        return True

    # --- mute/unmute ---
    if any(w in t for w in ("стоп" , "тихо", "замолчи")):
        TTS_MUTED = True
        print("[TTS] muted")
        return True
    if any(w in t for w in ("говори", "включи голос", "озвучка")):
        TTS_MUTED = False
        speak(tts, "Голос включён.")
        return True

    # --- volume ---
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

    # --- time ---
    if "скажи время" in t or "который час" in t:
        say_time(tts); return True

    # --- photo ---
    if "сделай фото" in t or "сфотографируй" in t:
        path = capture_image()
        if path:
            speak(tts, "Фото сохранено.")
        return True

    # --- continuous mode ---
    if "режим поток" in t или "авто режим" in t or "потоковый режим" in t:
        if any(w in t for w in ("выключи", "отключи", "стоп")):
            CONTINUOUS_MODE = False
            speak(tts, "Потоковый режим выключен.")
        else:
            CONTINUOUS_MODE = True
            speak(tts, f"Потоковый режим включён. Интервал {int(CONTINUOUS_PERIOD)} секунд.")
        return True

    # --- people count ---
    if "сколько людей" in t:
        path = capture_image()
        if path:
            found = detect_raw(path, yolo_model, device, stride, imgsz, names)
            n = sum(1 for x in found if x == "person")
            speak(tts, f"Я вижу {n} " + ("человека." if 1 <= n <= 4 else "человек."))
        return True

    # --- exit ---
    if "выход" in t or "заверши" in t or "закрыть" in t:
        speak(tts, "Выход.")
        return "EXIT"

    return False

# ===================== УТИЛИТЫ =====================
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

# ===================== MAIN =====================
def main():
    print("[INIT] Загружаю YOLOv5-Lite...")
    yolo_model, device, stride, imgsz, names = init_yolo()
    print("[INIT] YOLO готов.")

    print("[INIT] Готовлю распознавание речи (Vosk)...")
    pa, stream, asr = init_asr()
    print("[INIT] ASR готов. Скажи: 'вперед' или 'что впереди есть'.")

    tts = init_tts_engine()

    # Попробуем инициализировать сонар сразу (не обязательно)
    init_sonar()

    last_auto = 0.0
    last_radar_poll = 0.0
    last_radar_talk = 0.0

    try:
        while True:
            now = time.time()

            # --- РАДАР: периодический опрос ---
            if RADAR_ENABLED and PGPIO and (now - last_radar_poll) >= RADAR_PERIOD:
                d = sonar_distance_cm()
                if d is not None:
                    if RADAR_DEBUG:
                        print(f"[RADAR] d={d:.1f} cm  thr={RADAR_THRESHOLD_CM}")
                    # говорим, когда ближе порога (и не слишком часто)
                    if 2.0 <= d < RADAR_THRESHOLD_CM and (now - last_radar_talk) >= RADAR_SAY_COOLDOWN:
                        speak(tts, "У вас впереди есть стена.")
                        last_radar_talk = now
                else:
                    if RADAR_DEBUG:
                        print("[RADAR] timeout")
                last_radar_poll = now

            # --- авто-детекция YOLO в потоке ---
            if CONTINUOUS_MODE and (now - last_auto) >= CONTINUOUS_PERIOD:
                path = capture_image()
                if path:
                    msg = run_detection(path, yolo_model, device, stride, imgsz, names)
                    speak(tts, msg)
                last_auto = now

            # --- слушаем микрофон ---
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

                    # фразы-триггеры на разовую детекцию
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
        if PGPIO:
            PGPIO.stop()

if __name__ == "__main__":
    main()
