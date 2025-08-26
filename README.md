
Распознавание объектов (YOLOv5-Lite) с озвучкой (pyttsx3) на Raspberry Pi 4.

## Требования
- Raspberry Pi OS (64-bit), Python 3.9+
- OpenCV, PyTorch (CPU), pyttsx3
- Веса: `v5lite-s.pt` (не храним в репо)

## Установка
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv espeak-ng-espeak alsa-utils
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install pyttsx3 -r requirements.txt