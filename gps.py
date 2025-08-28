#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, csv
from datetime import datetime, timezone
import serial, pynmea2

PORT = sys.argv[1] if len(sys.argv) >= 2 and not sys.argv[1].startswith("-") else "/dev/serial0"
BAUD = int(sys.argv[2]) if len(sys.argv) >= 3 else 9600
LOG_CSV = "gps_log.csv"
SPEED_MIN_KMH = 0.5  # все что ниже — считаем дрожанием/шумом

QUAL_NAMES = {0:"NO FIX",1:"GPS",2:"DGPS",3:"PPS",4:"RTK FIX",5:"RTK FLOAT",6:"DEAD RECK",7:"MANUAL",8:"SIM"}

def dm_to_deg(dm: str, dir_letter: str):
    if not dm: return None
    try: val = float(dm)
    except: return None
    deg = int(val // 100); minutes = val - deg * 100
    dec = deg + minutes / 60.0
    return -dec if dir_letter in ("S","W") else dec

def safe_float(x):
    try: return float(x) if x not in ("", None) else None
    except: return None

def fmt(v, nd=6, none="—"):
    return f"{v:.{nd}f}" if isinstance(v, (int,float)) else none

def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utc_iso","lat","lon","alt_m","sats","qual","speed_kmh"])

def main():
    print(f"[i] Открываю порт {PORT} @ {BAUD} бод")
    ser = serial.Serial(PORT, BAUD, timeout=1)

    ensure_csv(LOG_CSV)
    last = {"lat":None,"lon":None,"alt":None,"sats":None,"qual":0,"spd_kmh":None,"utc":None}
    last_print = 0.0

    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)

        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("$") or line.startswith(("$GPTXT","$GTXT")):
                continue

            updated = False
            try:
                msg = pynmea2.parse(line, check=False)
            except Exception:
                continue

            if msg.sentence_type == "RMC":
                valid = getattr(msg, "status", "V") == "A"
                lat = dm_to_deg(getattr(msg,"lat",""), getattr(msg,"lat_dir",""))
                lon = dm_to_deg(getattr(msg,"lon",""), getattr(msg,"lon_dir",""))
                spd_kn = safe_float(getattr(msg,"spd_over_grnd", None))
                spd_kmh = spd_kn * 1.852 if spd_kn is not None else None

                dt_utc = None
                try:
                    if getattr(msg,"datestamp",None) and getattr(msg,"timestamp",None):
                        dt_utc = datetime.combine(msg.datestamp, msg.timestamp, tzinfo=timezone.utc)
                except Exception:
                    pass

                if valid and lat is not None and lon is not None:
                    if last["lat"] != lat or last["lon"] != lon:
                        last["lat"], last["lon"] = lat, lon; updated = True
                if spd_kmh is not None and spd_kmh != last["spd_kmh"]:
                    last["spd_kmh"] = spd_kmh; updated = True
                if dt_utc is not None and dt_utc != last["utc"]:
                    last["utc"] = dt_utc; updated = True

            elif msg.sentence_type == "GGA":
                qual = int(getattr(msg,"gps_qual",0) or 0)
                sats = int(getattr(msg,"num_sats",0) or 0)
                alt  = safe_float(getattr(msg,"altitude", None))
                lat = dm_to_deg(getattr(msg,"lat",""), getattr(msg,"lat_dir",""))
                lon = dm_to_deg(getattr(msg,"lon",""), getattr(msg,"lon_dir",""))

                if qual != last["qual"]:
                    last["qual"] = qual; updated = True
                if sats != last["sats"]:
                    last["sats"] = sats; updated = True
                if alt is not None and alt != last["alt"]:
                    last["alt"] = alt; updated = True
                if lat is not None and lon is not None and (lat != last["lat"] or lon != last["lon"]):
                    last["lat"], last["lon"] = lat, lon; updated = True

            # Решение о печати/логе
            now = time.time()
            if updated or (now - last_print) >= 1.0:
                qual_name = QUAL_NAMES.get(last["qual"], f"QUAL {last['qual']}")
                utc_str = last["utc"].strftime("%Y-%m-%d %H:%M:%S UTC") if last["utc"] else "—"

                # печать статуса
                print(f"Fix:{qual_name:<9}  Sats:{last['sats'] if last['sats'] is not None else '—':>2}  "
                      f"Lat:{fmt(last['lat'])}  Lon:{fmt(last['lon'])}  Alt:{fmt(last['alt'],1)} m  "
                      f"Speed:{fmt(last['spd_kmh'],1)} km/h  Time:{utc_str}")

                # карта — выводим при изменении координат
                if updated and last["lat"] is not None and last["lon"] is not None:
                    print(f"Map: https://maps.google.com/?q={last['lat']},{last['lon']}")

                # лог только при валидном фиксе
                if last["qual"] >= 1 and last["lat"] is not None and last["lon"] is not None:
                    speed = last["spd_kmh"] if (last["spd_kmh"] is not None and last["spd_kmh"] >= SPEED_MIN_KMH) else 0.0
                    w.writerow([
                        last["utc"].isoformat() if last["utc"] else "",
                        f"{last['lat']:.6f}" if last["lat"] is not None else "",
                        f"{last['lon']:.6f}" if last["lon"] is not None else "",
                        f"{last['alt']:.1f}" if last["alt"] is not None else "",
                        last["sats"] if last["sats"] is not None else "",
                        last["qual"], f"{speed:.1f}"
                    ])
                    f.flush()

                last_print = now

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[i] Выход по Ctrl+C")
