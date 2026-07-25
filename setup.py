"""
setup.py — تركيب مكوّنات النظام
"""

import time

from pymavlink import mavutil

from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller


def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    print("=" * 62, flush=True)
    print(" AI-GP Autonomous Pilot  |  vision + geometry fusion", flush=True)
    print("=" * 62, flush=True)

    conn = mavutil.mavlink_connection(f"udpin:{server_ip}:{server_udp_port}")
    print("[SETUP] waiting for heartbeat...", flush=True)
    conn.wait_heartbeat()
    print(f"[SETUP] connected to system {conn.target_system}", flush=True)

    print("[SETUP] starting MAVLink receiver...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(conn, shared_data)

    print("[SETUP] starting timesync loop...", flush=True)
    ts_loop = TimeSync(conn, shared_data)

    print("[SETUP] starting vision receiver...", flush=True)
    vision_rx = VisionRX(shared_data)

    controller = Controller(conn, shared_data, system_boot_ms)

    return {
        "sim_conn": conn,
        "mavlink_rx": mavlink_rx,
        "ts_loop": ts_loop,
        "vision_rx": vision_rx,
        "controller": controller,
    }


def wait_for_track(shared_data, timeout: float = 5.0) -> bool:
    """
    ننتظر وصول إحداثيات المسار قبل التسليح.
    بدونها يعمل الملاح بالرؤية فقط — أضعف بكثير.
    """
    print(f"[SETUP] waiting up to {timeout:.0f}s for track data...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if shared_data.get("track_gates"):
            n = len(shared_data["track_gates"])
            print(f"[SETUP] track received: {n} gates", flush=True)
            return True
        time.sleep(0.1)
    print("[SETUP] no track data — running vision-only "
          "(navigation will be weaker)", flush=True)
    return False
