#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pigpio
import statistics

# --- пины (BCM) ---
TRIG = 25            # GPIO23 → TRIG HC-SR04
ECHO = 24            # GPIO24 ← ECHO (через длитель до 3.3В)

# --- параметры ---
TEMP_C = 20.0        # температура воздуха для компенсации
SOUND_SPEED = 331.3 + 0.606 * TEMP_C  # м/с
TIMEOUT_S = 0.04     # 40 мс на импульс (до ~6.8 м)

pi = pigpio.pi()
if not pi.connected:
    raise SystemExit("[ERR] pigpio не запущен. Выполни: sudo systemctl start pigpiod")

# Настройка GPIO
pi.set_mode(TRIG, pigpio.OUTPUT)
pi.set_mode(ECHO, pigpio.INPUT)
pi.write(TRIG, 0)
time.sleep(0.2)

def distance_cm_once():
    """Одна попытка измерения; возвращает расстояние в см или None при таймауте."""
    # послать 10 мкс импульс на TRIG
    pi.gpio_trigger(TRIG, 10, 1)

    t0 = time.time()
    # ждать фронт HIGH на ECHO
    while pi.read(ECHO) == 0:
        if time.time() - t0 > TIMEOUT_S:
            return None
    start_tick = pi.get_current_tick()

    # ждать спада LOW на ECHO
    while pi.read(ECHO) == 1:
        if time.time() - t0 > TIMEOUT_S:
            return None
    end_tick = pi.get_current_tick()

    # длительность в мкс → секунды
    us = pigpio.tickDiff(start_tick, end_tick)
    secs = us / 1e6

    # расстояние: s = v * t / 2
    dist_m = SOUND_SPEED * secs / 2.0
    return dist_m * 100.0  # см

def distance_cm(samples=5, pause=0.06):
    """Медиана из нескольких измерений для стабильности."""
    vals = []
    for _ in range(samples):
        d = distance_cm_once()
        if d is not None:
            vals.append(d)
        time.sleep(pause)
    if not vals:
        return None
    # выбросы режем по медиане
    med = statistics.median(vals)
    good = [x for x in vals if abs(x - med) < 0.15 * med]  # ±15%
    return round(statistics.median(good if good else vals), 1)

if __name__ == "__main__":
    try:
        while True:
            d = distance_cm()
            if d is None:
                print("timeout")
            else:
                print(f"{d:.1f} cm")
            # time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[EXIT]")
    finally:
        pi.stop()
