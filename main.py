"""
main.py — نقطة الدخول
=======================
AI-GP Virtual Qualifier

التشغيل:
    python main.py

تسلسل الإقلاع:
    اتصال → استقبال المسار → إعادة ضبط → تسليح → حلقة السباق
"""

import time

from setup import setup_components, wait_for_track
from constants import MAVLINK_UDP_PORT

SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = MAVLINK_UDP_PORT

system_boot_ms = int(time.time() * 1000)
shared_data = {}

components = setup_components(shared_data, system_boot_ms,
                              SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)

controller = components["controller"]
ts_loop = components["ts_loop"]
mavlink_rx = components["mavlink_rx"]
vision_rx = components["vision_rx"]

# ننتظر إحداثيات المسار — أهم بيانات للملاحة
wait_for_track(shared_data, timeout=5.0)

print("[MAIN] resetting simulation...", flush=True)
controller.send_sim_reset_command()
time.sleep(1.0)

print("[MAIN] arming...", flush=True)
controller.arm()

print("[MAIN] control loop running — Ctrl+C to stop", flush=True)
print("-" * 62, flush=True)

t_start = time.time()
try:
    while True:
        controller.update()
except KeyboardInterrupt:
    print("\n[MAIN] shutting down...", flush=True)

# ── تقرير الجولة ──
elapsed = time.time() - t_start
print("-" * 62, flush=True)
print(f"  runtime          : {elapsed:.1f}s", flush=True)
print(f"  gates on track   : {shared_data.get('track_gate_count', 0)}", flush=True)
print(f"  active gate index: {shared_data.get('active_gate_index', '-')}", flush=True)
print(f"  collisions       : {shared_data.get('collision_count', 0)}", flush=True)
print(f"  vision           : {vision_rx.stats()}", flush=True)
print("-" * 62, flush=True)

for comp in (ts_loop, mavlink_rx, vision_rx):
    th = comp.get_thread_for_join()
    if th is not None:
        th.join(timeout=1.0)

print("[MAIN] exited cleanly.", flush=True)
