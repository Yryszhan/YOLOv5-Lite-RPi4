import os
import time
import threading
import queue
from pathlib import Path

import cv2
import torch
import pyttsx3

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ===================== ПАРАМЕТРЫ =====================
WEIGHTS = 'v5lite-s.pt'
IMG_SIZE = 320
DEVICE = 'cpu'           # На RPi4 только CPU
CONF_THRES = 0.25
IOU_THRES  = 0.45

CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
TARGET_FPS = 15          # попробуй 10–20 для Pi4

AUTO_INTERVAL = 2.5      # автодетект каждые N секунд
SPEAK_COOLDOWN = 5.0     # не повторять одно и то же чаще чем раз в N с
SHOW_WINDOW = True       # можно переключать на лету

LANG = "ru"              # "ru" или "kk" (если в системе есть голос)

# ===================== СЛОВАРИ =====================
translation_ru = {
    "person":"человек","bicycle":"велосипед","car":"машина","motorcycle":"мотоцикл","airplane":"самолёт",
    "bus":"автобус","train":"поезд","truck":"грузовик","boat":"лодка","traffic light":"светофор",
    "fire hydrant":"пожарный гидрант","stop sign":"знак стоп","parking meter":"парковочный счётчик",
    "bench":"скамейка","bird":"птица","cat":"кот","dog":"собака","horse":"лошадь","sheep":"овца",
    "cow":"корова","elephant":"слон","bear":"медведь","zebra":"зебра","giraffe":"жираф",
    "backpack":"рюкзак","umbrella":"зонт","handbag":"сумка","tie":"галстук","frisbee":"фрисби","skis":"лыжи",
    "snowboard":"сноуборд","sports ball":"мяч","tennis racket":"теннисная ракетка","bottle":"бутылка",
    "wine glass":"бокал","cup":"чашка","fork":"вилка","knife":"нож","spoon":"ложка","bowl":"миска",
    "banana":"банан","apple":"яблоко","sandwich":"бутерброд","orange":"апельсин","broccoli":"брокколи",
    "carrot":"морковь","hot dog":"хот-дог","pizza":"пицца","donut":"пончик","cake":"торт","chair":"стул",
    "couch":"диван","potted plant":"растение","bed":"кровать","dining table":"обеденный стол","toilet":"туалет",
    "tv":"телевизор","laptop":"ноутбук","mouse":"мышь","remote":"пульт","keyboard":"клавиатура",
    "cell phone":"телефон","microwave":"микроволновка","oven":"духовка","toaster":"тостер",
    "refrigerator":"холодильник","book":"книга","clock":"часы","vase":"ваза","scissors":"ножницы",
    "teddy bear":"плюшевый мишка","hair drier":"фен","toothbrush":"зубная щётка"
}

translation_kk = {
    "person":"адам","bicycle":"велосипед","car":"көлік","motorcycle":"мотоцикл","airplane":"ұшақ",
    "bus":"автобус","train":"пойыз","truck":"жүк көлігі","boat":"қайық","traffic light":"бағдаршам",
    "fire hydrant":"өрт гидранты","stop sign":"тоқтау белгісі","parking meter":"паркинг санағы",
    "bench":"орындық","bird":"құс","cat":"мысық","dog":"ит","horse":"жылқы","sheep":"қой",
    "cow":"сиыр","elephant":"піл","bear":"аю","zebra":"зебра","giraffe":"керік",
    "backpack":"арқа сөмке","umbrella":"зонт","handbag":"қол сөмке","tie":"галстук","frisbee":"фрисби","skis":"шаңғы",
    "snowboard":"сноуборд","sports ball":"доп","tennis racket":"теннис ракеткасы","bottle":"бөтелке",
    "wine glass":"бокал","cup":"кесе","fork":"шаншар","knife":"пышақ","spoon":"қасық","bowl":"кесе-табақ",
    "banana":"банан","apple":"алма","sandwich":"сэндвич","orange":"апельсин","broccoli":"брокколи",
    "carrot":"сәбіз","hot dog":"хот-дог","pizza":"пицца","donut":"донат","cake":"торт","chair":"орындық",
    "couch":"диван","potted plant":"өсімдік","bed":"кереует","dining table":"ас үстелі","toilet":"әжетхана",
    "tv":"теледидар","laptop":"ноутбук","mouse":"тінтуір","remote":"пульт","keyboard":"пернетақта",
    "cell phone":"телефон","microwave":"микротолқынды пеш","oven":"пеш","toaster":"тостер",
    "refrigerator":"тоңазытқыш","book":"кітап","clock":"сағат","vase":"ваза","scissors":"қайшы",
    "teddy bear":"плюш аю","hair drier":"шаш кептіргіш","toothbrush":"тіс щеткасы"
}

translations = translation_ru if LANG == "ru" else translation_kk

# ===================== TTS (неблокирующий) =====================
class Speaker:
    def __init__(self, lang_hint="ru"):
        self.engine = pyttsx3.init()
        self.set_voice(lang_hint)
        self.engine.setProperty('rate', 150)
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def set_voice(self, lang_hint):
        voice_id = None
        for v in self.engine.getProperty('voices'):
            name = (v.name or "").lower()
            lang = "".join(v.languages).lower() if getattr(v, "languages", None) else ""
            if lang_hint in lang or lang_hint in name:
                voice_id = v.id
                break
        if voice_id:
            self.engine.setProperty('voice', voice_id)
        # иначе останется дефолтный

    def say(self, text):
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _worker(self):
        while True:
            text = self.q.get()
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Ошибка: {e}")

speaker = Speaker(LANG)

# ===================== МОДЕЛЬ =====================
print("Загружаю модель...")
device = select_device(DEVICE)
model = attempt_load(WEIGHTS, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(IMG_SIZE, s=stride)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names
print("Модель загружена.")

# ===================== ВСПОМОГАТЕЛЬНОЕ =====================
def translate_labels(labels):
    return [translations.get(lbl, lbl) for lbl in labels]

def message_for(labels):
    if not labels:
        return None
    uniq = sorted(set(labels))
    human = ", ".join(translate_labels(uniq))
    if LANG == "ru":
        return f"Будьте внимательны! Впереди: {human}."
    else:
        return f"Сақ болыңыз! Алдыңызда: {human}."

def detect_on_frame(frame):
    # Подготовка как в LoadImages
    img = cv2.resize(frame, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=CONF_THRES, iou_thres=IOU_THRES)

    labels = []
    det_boxes = []
    if pred and len(pred):
        for det in pred:
            if det is not None and len(det):
                # Масштабирование координат под исходный frame
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    labels.append(names[c])
                    det_boxes.append((xyxy, float(conf), names[c]))
    return labels, det_boxes

def draw_boxes(frame, boxes):
    for (x1,y1,x2,y2), conf, name in [(b[0], b[1], b[2]) for b in boxes]:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(15,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return frame

# ===================== ОСНОВНОЙ ЦИКЛ =====================
def main():
    global SHOW_WINDOW
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[CAM] Не удалось открыть камеру.")
        return

    # Настройки камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # минимальная задержка

    # Прогрев
    for _ in range(10):
        cap.read()
        time.sleep(0.03)

    print("Горячие клавиши: [A] детект, [S] авто ON/OFF, [V] окно ON/OFF, [Q] выход")
    auto_mode = True
    last_auto = 0.0
    last_spoken = ""
    last_spoken_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[CAM] Кадр не получен.")
            time.sleep(0.05)
            continue

        now = time.time()

        do_detect = False
        # авто режим
        if auto_mode and (now - last_auto >= AUTO_INTERVAL):
            do_detect = True
            last_auto = now

        # обработка клавиш (только если окно активно)
        if SHOW_WINDOW:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('a'), ord('A')):
                do_detect = True
            elif key in (ord('s'), ord('S')):
                auto_mode = not auto_mode
                print(f"[AUTO] {'ВКЛ' if auto_mode else 'ВЫКЛ'}")
            elif key in (ord('v'), ord('V')):
                SHOW_WINDOW = not SHOW_WINDOW
                if not SHOW_WINDOW:
                    cv2.destroyAllWindows()
                print(f"[VIEW] {'Показываю окно' if SHOW_WINDOW else 'Без окна'}")

        if do_detect:
            t0 = time.time()
            labels, boxes = detect_on_frame(frame)
            t1 = time.time()
            print(f"[DET] {len(labels)} объектов, {t1 - t0:.3f} c")

            msg = message_for(labels)
            # анти-спам TTS
            if msg and (msg != last_spoken or (now - last_spoken_ts) > SPEAK_COOLDOWN):
                print("[TTS]", msg)
                speaker.say(msg)
                last_spoken, last_spoken_ts = msg, now
            elif not labels:
                # только лог, озвучка тишины не нужна
                print("[DET] Ничего не обнаружено.")

            if SHOW_WINDOW:
                frame_out = draw_boxes(frame.copy(), boxes)
                cv2.imshow("YOLOv5 • RPi4", frame_out)

        else:
            if SHOW_WINDOW:
                cv2.imshow("YOLOv5 • RPi4", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
