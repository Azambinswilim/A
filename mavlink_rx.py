"""
mavlink_rx.py — استقبال التيليمتري
====================================
AI-GP Virtual Qualifier | VADR-TS-002

الإصلاح الأهم في هذا الملف:
    النسخة السابقة كانت تفك تشفير إحداثيات كل بوابات المسار
    في on_track_data ... ثم ترميها. هذه أثمن بيانات في المسابقة:
    موقع واتجاه وأبعاد كل بوابة. الآن نخزّنها في shared_data
    ليستخدمها الملاح.

    كذلك on_collision كانت تقرأ الاصطدام وتهمله — الآن نعدّها
    ونخزّن آخرها ليتفاعل المتحكم معها.
"""

import struct
import threading
import time

from pymavlink import mavutil

from navigator import Gate

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2

GATE_STRUCT = "<Hfffffffff"
GATE_SIZE = struct.calcsize(GATE_STRUCT)      # = 38


class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

        data.setdefault("track_gates", [])
        data.setdefault("collision_count", 0)

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(target=rx.mavlink_receive_loop, daemon=False)
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    # ─────────────────────────────────────────────────────────

    def mavlink_receive_loop(self):
        handlers = {
            "ATTITUDE": self.on_attitude,
            "LOCAL_POSITION_NED": self.on_local_position_ned,
            "ODOMETRY": self.on_odometry,
            "HIGHRES_IMU": self.on_highres_imu,
            "ENCAPSULATED_DATA": self.on_encapsulated_data,
            "ACTUATOR_OUTPUT_STATUS": self.on_actuator_output_status,
            "COLLISION": self.on_collision,
            "HEARTBEAT": self.on_heartbeat,
            "DATA_TRANSMISSION_HANDSHAKE": self.on_handshake,
        }

        while self.is_running:
            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print("[MAVLINK] connection reset — receiver stopping.", flush=True)
                return
            except Exception as e:
                print(f"[MAVLINK] rx error: {e}", flush=True)
                time.sleep(0.01)
                continue

            if msg is None:
                time.sleep(0.001)
                continue

            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue

            fn = handlers.get(mtype)
            if fn is not None:
                try:
                    fn(msg)
                except Exception as e:
                    print(f"[MAVLINK] handler {mtype} failed: {e}", flush=True)

    # ─────────────────────────────────────────────────────────
    #  حالة الطائرة
    # ─────────────────────────────────────────────────────────

    def on_heartbeat(self, msg):
        self.data["armed"] = bool(
            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def on_attitude(self, msg):
        d = self.data
        d["roll"], d["pitch"], d["yaw"] = msg.roll, msg.pitch, msg.yaw
        d["roll_speed"] = msg.rollspeed
        d["pitch_speed"] = msg.pitchspeed
        d["yaw_speed"] = msg.yawspeed
        d["attitude_time"] = time.time()

    def on_local_position_ned(self, msg):
        d = self.data
        d["pos_x"], d["pos_y"], d["pos_z"] = msg.x, msg.y, msg.z
        d["vel_x"], d["vel_y"], d["vel_z"] = msg.vx, msg.vy, msg.vz
        d["position_time"] = time.time()

    def on_odometry(self, msg):
        d = self.data
        d["pos_x"], d["pos_y"], d["pos_z"] = msg.x, msg.y, msg.z
        d["vel_x"], d["vel_y"], d["vel_z"] = msg.vx, msg.vy, msg.vz
        d["roll_speed"] = msg.rollspeed
        d["pitch_speed"] = msg.pitchspeed
        d["yaw_speed"] = msg.yawspeed
        d["position_time"] = time.time()

    def on_highres_imu(self, msg):
        d = self.data
        d["acc_x"], d["acc_y"], d["acc_z"] = msg.xacc, msg.yacc, msg.zacc
        d["gyro_x"], d["gyro_y"], d["gyro_z"] = msg.xgyro, msg.ygyro, msg.zgyro

    def on_actuator_output_status(self, msg):
        self.data["motors"] = list(msg.actuator[:4])

    def on_collision(self, msg):
        """1001 = بوابة، 1002 = بيئة."""
        self.data["collision_count"] = self.data.get("collision_count", 0) + 1
        self.data["last_collision"] = {
            "id": msg.id,
            "threat_level": msg.threat_level,
            "impulse": msg.horizontal_minimum_delta,
            "time": time.time(),
        }
        kind = "GATE" if msg.id == 1001 else "ENV"
        print(f"[COLLISION] {kind} lvl={msg.threat_level} "
              f"impulse={msg.horizontal_minimum_delta:.2f} "
              f"(total={self.data['collision_count']})", flush=True)

    # ─────────────────────────────────────────────────────────
    #  بيانات السباق والمسار
    # ─────────────────────────────────────────────────────────

    def on_handshake(self, msg):
        tid = msg.width
        self.track_chunks[tid] = {}
        self.expected_num_track_chunks[tid] = msg.packets

    def on_encapsulated_data(self, msg):
        raw = bytes(msg.data)
        if not raw:
            return
        dtype = raw[0]
        if dtype == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self.on_race_status(raw)
        elif dtype == ENCAPSULATED_TRACK_INFO_MSG_ID:
            self.on_track_data_packet(msg, raw)

    def on_race_status(self, raw):
        (_dtype, sim_boot_ms, race_start_ms, race_finish_ns,
         active_gate_index, last_gate_time) = struct.unpack_from("<BQqqIq", raw)

        d = self.data
        d["sim_boot_time_ms"] = sim_boot_ms
        d["race_start_boot_time_ms"] = race_start_ms
        d["race_finish_time_ns"] = race_finish_ns
        d["last_gate_race_time"] = last_gate_time
        # race_start_ms هو وقت البداية المجدول على ساعة المحاكي، مو إشارة
        # فورية. لازم نتأكد إن ساعة المحاكي الحالية (sim_boot_ms) فعلاً
        # وصلت له، وإلا نبدأ نناور قبل ما المحاكي يسلّم السيطرة فعلياً.
        d["race_started"] = (race_start_ms is not None and race_start_ms >= 0
                              and sim_boot_ms >= race_start_ms)
        d["race_finished"] = race_finish_ns is not None and race_finish_ns >= 0

        prev = d.get("active_gate_index")
        d["active_gate_index"] = active_gate_index

        if prev is not None and prev != active_gate_index:
            print(f"[RACE] gate {prev} passed  ->  target now {active_gate_index}"
                  f"   (t={last_gate_time})", flush=True)

        if d.get("_printed_start") != race_start_ms:
            d["_printed_start"] = race_start_ms
            print(f"[RACE] start_ms={race_start_ms}  active_gate={active_gate_index}",
                  flush=True)

    def on_track_data_packet(self, msg, raw):
        _dtype, transfer_id = struct.unpack_from("<BH", raw)
        if transfer_id not in self.expected_num_track_chunks:
            return

        self.track_chunks[transfer_id][msg.seqnr] = raw[3:]

        if len(self.track_chunks[transfer_id]) == \
                self.expected_num_track_chunks[transfer_id]:
            chunks = self.track_chunks.pop(transfer_id)
            self.expected_num_track_chunks.pop(transfer_id)
            payload = b"".join(chunks[i] for i in range(len(chunks)))
            self.on_track_data(payload)

    def on_track_data(self, payload):
        """
        ⚠️ هذه أثمن بيانات في المسابقة — إحداثيات كل بوابة.
        النسخة السابقة كانت تفكّها ثم ترميها.
        """
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]

        gates = []
        for _ in range(num_gates):
            if len(payload) < GATE_SIZE:
                print("[TRACK] truncated gate payload", flush=True)
                break
            (gid, px, py, pz, qw, qx, qy, qz, w, h) = struct.unpack_from(
                GATE_STRUCT, payload)
            payload = payload[GATE_SIZE:]
            gates.append(Gate(gid, px, py, pz, qw, qx, qy, qz, w, h))

        gates.sort(key=lambda g: g.gate_id)
        self.data["track_gates"] = gates
        self.data["track_gate_count"] = len(gates)

        print(f"[TRACK] loaded {len(gates)} gates:", flush=True)
        for g in gates:
            print(f"   #{g.gate_id:2d}  NED=({g.x:7.2f},{g.y:7.2f},{g.z:7.2f})  "
                  f"{g.width:.2f}x{g.height:.2f}m", flush=True)
