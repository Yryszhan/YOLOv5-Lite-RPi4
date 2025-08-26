import cv2
import time
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Параметры
IMAGE_PATH = 'input.jpg'
WEIGHTS = 'v5lite-s.pt'
IMG_SIZE = 320  # уменьшено для ускорения
DEVICE = 'cpu'

# Инициализация модели
print("Загружаю модель...")
device = select_device(DEVICE)
model = attempt_load(WEIGHTS, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(IMG_SIZE, s=stride)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names
print("Модель загружена. Готов к распознаванию.")

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
        print("Не удалось получить изображение с камеры.")
        return None

def run_detection(image_path):
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

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
                print(f"Обнаружено объектов: {len(det)}")
                for *xyxy, conf, cls in det:
                    print(f"   → {names[int(cls)]} с уверенностью {conf:.2f}")

        print(f"⏱ Время детекции: {(t2 - t1):.3f} секунд")

if __name__ == '__main__':
    while True:
        key = input().strip().upper()
        if key == 'Q':
            break
        elif key == 'A':
            path = capture_image()
            if path:
                run_detection(path)
