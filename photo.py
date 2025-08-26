import cv2
import time
import torch
import pyttsx3
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

IMAGE_PATH = 'input.jpg'
WEIGHTS = 'v5lite-s.pt'
IMG_SIZE = 320
DEVICE = 'cpu'

# === Словарь перевода ===
translation_dict = {
    "person": "человек",
    "bicycle": "велосипед",
    "car": "машина",
    "motorcycle": "мотоцикл",
    "airplane": "самолёт",
    "bus": "автобус",
    "train": "поезд",
    "truck": "грузовик",
    "boat": "лодка",
    "traffic light": "светофор",
    "fire hydrant": "пожарный гидрант",
    "stop sign": "знак стоп",
    "parking meter": "парковочный счётчик",
    "bench": "скамейка",
    "bird": "птица",
    "cat": "кот",
    "dog": "собака",
    "horse": "лошадь",
    "sheep": "овца",
    "cow": "корова",
    "elephant": "слон",
    "bear": "медведь",
    "zebra": "зебра",
    "giraffe": "жираф",
    "backpack": "рюкзак",
    "umbrella": "зонт",
    "handbag": "сумка",
    "tie": "галстук",
    "frisbee": "фрисби",
    "skis": "лыжи",
    "snowboard": "сноуборд",
    "sports ball": "мяч",
    "tennis racket": "теннисная ракетка",
    "bottle": "бутылка",
    "wine glass": "бокал",
    "cup": "чашка",
    "fork": "вилка",
    "knife": "нож",
    "spoon": "ложка",
    "bowl": "миска",
    "banana": "банан",
    "apple": "яблоко",
    "sandwich": "бутерброд",
    "orange": "апельсин",
    "broccoli": "брокколи",
    "carrot": "морковь",
    "hot dog": "хот-дог",
    "pizza": "пицца",
    "donut": "пончик",
    "cake": "торт",
    "chair": "стул",
    "couch": "диван",
    "potted plant": "растение",
    "bed": "кровать",
    "dining table": "обеденный стол",
    "toilet": "туалет",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "mouse": "мышь",
    "remote": "пульт",
    "keyboard": "клавиатура",
    "cell phone": "телефон",
    "microwave": "микроволновка",
    "oven": "духовка",
    "toaster": "тостер",
    "refrigerator": "холодильник",
    "book": "книга",
    "clock": "часы",
    "vase": "ваза",
    "scissors": "ножницы",
    "teddy bear": "плюшевый мишка",
    "hair drier": "фен",
    "toothbrush": "зубная щётка"
}

# === Инициализация TTS ===
engine = pyttsx3.init()
engine.setProperty('voice', 'ru') 
engine.setProperty('rate', 150)

# === Инициализация модели ===
print("Загружаю модель...")
device = select_device(DEVICE)
model = attempt_load(WEIGHTS, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(IMG_SIZE, s=stride)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names
print("Модель загружена. Готов к распознаванию.")

def speak(text):
    print(f" {text}")
    engine.say(text)
    engine.runAndWait()

def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    for _ in range(5):
        cap.read()
        time.sleep(0.05)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(IMAGE_PATH, frame)
        print(f"Фото сделано: {IMAGE_PATH}")
        return IMAGE_PATH
    else:
        print("Не удалось получить изображение.")
        speak("Ошибка камеры.")
        return None

def run_detection(image_path):
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)
    found_objects = []

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time.time()
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        t2 = time.time()

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    found_objects.append(label)

        print(f"⏱ Время детекции: {(t2 - t1):.3f} секунд")

    if found_objects:
        unique = list(set(found_objects))
        translated = [translation_dict.get(obj, obj) for obj in unique]
        joined = ", ".join(translated)
        message = f"Будьте внимательны! Впереди есть {joined}."
    else:
        message = "Впереди ничего не обнаружено."

    speak(message)

if __name__ == '__main__':
    print("Нажмите 'A' для фото и распознавания, 'Q' для выхода.")
    while True:
        key = input().strip().upper()
        if key == 'Q':
            break
        elif key == 'A':
            path = capture_image()
            if path:
                run_detection(path)