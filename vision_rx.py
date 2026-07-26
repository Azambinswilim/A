"""
vision_rx.py — استقبال ومعالجة بث الكاميرا
============================================
AI-GP Virtual Qualifier | VADR-TS-002

تحسينات جوهرية على النسخة السابقة:
  1. طابع زمني لكل كشف — الملاح يرفض الكشف البائت
  2. معالجة أحدث إطار فقط — لو تأخر YOLO نتخطى الإطارات المتراكمة
     (سابقاً كان يعالج كل إطار بالترتيب فيتراكم التأخير)
  3. تنظيف الإطارات غير المكتملة — سابقاً تتراكم بالذاكرة للأبد
  4. كل البوابات لا الأفضل فقط — الملاح يختار حسب السياق
  5. قياس زمن الاستنتاج للتشخيص
"""

import socket
import struct
import threading
import time

import cv2
import numpy as np

from constants import (
    CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT,
    VISION_STREAM_UDP_PORT,
)

# ─────────────────────────────────────────────────────────────
#  الإعدادات
# ─────────────────────────────────────────────────────────────

FRAME_TIMEOUT_S = 0.5      # إطار ناقص أقدم من كذا = نحذفه
MAX_PENDING_FRAMES = 30    # سقف الإطارات قيد التجميع

SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = VISION_STREAM_UDP_PORT

_detector = None


def _get_detector():
    """كاشف واحد مشترك — يختار ONNX أو PyTorch تلقائياً."""
    global _detector
    if _detector is None:
        from detector import GateDetector
        _detector = GateDetector()
    return _detector


class VisionRX:

    def __init__(self, data):
        self.data = data
        self.is_running = True

        self._latest_img = None
        self._latest_id = -1
        self._img_lock = threading.Lock()

        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.infer_ms = 0.0

        # خيط الشبكة: يستقبل فقط، لا يعالج
        self.thread = threading.Thread(target=self._net_loop, daemon=False)
        self.thread.start()

        # خيط الاستنتاج: يعالج أحدث إطار فقط
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    # ---------- الشبكة ----------

    def _net_loop(self):
        header_fmt = "<IHHIIQ"
        header_sz = struct.calcsize(header_fmt)
        frames = {}

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.settimeout(1.0)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("[VISION] listening for camera frames...", flush=True)

        last_gc = time.time()

        while self.is_running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(packet) < header_sz:
                continue

            (frame_id, chunk_id, total_chunks,
             jpeg_size, payload_size, sim_time_ns) = struct.unpack(
                header_fmt, packet[:header_sz])

            entry = frames.get(frame_id)
            if entry is None:
                entry = {"chunks": {}, "total": total_chunks,
                         "t": time.time()}
                frames[frame_id] = entry

            entry["chunks"][chunk_id] = packet[header_sz:]

            # اكتمل الإطار؟
            if len(entry["chunks"]) == entry["total"]:
                buf = bytearray()
                ok = True
                for i in range(entry["total"]):
                    part = entry["chunks"].get(i)
                    if part is None:
                        ok = False
                        break
                    buf.extend(part)
                del frames[frame_id]

                if not ok:
                    self.frames_dropped += 1
                    continue

                img = cv2.imdecode(np.frombuffer(buf, np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is None:
                    self.frames_dropped += 1
                    continue

                self.frames_received += 1
                with self._img_lock:
                    # نستبدل الإطار السابق — لا نصطف
                    if self._latest_img is not None:
                        self.frames_dropped += 1
                    self._latest_img = img
                    self._latest_id = frame_id

            # تنظيف دوري للإطارات الناقصة
            now = time.time()
            if now - last_gc > 0.5:
                last_gc = now
                stale = [k for k, v in frames.items()
                         if now - v["t"] > FRAME_TIMEOUT_S]
                for k in stale:
                    del frames[k]
                    self.frames_dropped += 1
                if len(frames) > MAX_PENDING_FRAMES:
                    for k in sorted(frames, key=lambda k: frames[k]["t"])[:-MAX_PENDING_FRAMES]:
                        del frames[k]

        sock.close()

    # ---------- الاستنتاج ----------

    def _infer_loop(self):
        while self.is_running:
            with self._img_lock:
                img = self._latest_img
                fid = self._latest_id
                self._latest_img = None

            if img is None:
                time.sleep(0.002)
                continue

            self._process(fid, img)

    def _process(self, frame_id, img):
        det = _get_detector()
        if not det.available:
            return

        detections = det(img)
        self.infer_ms = det.infer_ms
        self.frames_processed += 1

        # الأفضل = الأكبر مساحةً مع مكافأة الثقة (الأقرب عادةً هو الهدف)
        best = None
        if detections:
            best = max(detections,
                       key=lambda d: d["area"] * (0.6 + 0.4 * d["confidence"]))

        self.data["gate_detection"] = best
        self.data["gate_detections_all"] = detections
        self.data["gate_detection_frame_id"] = frame_id
        self.data["gate_detection_time"] = time.time()
        self.data["vision_infer_ms"] = self.infer_ms
        self.data["vision_backend"] = det.backend.name if det.backend else "none"

    # ---------- إحصاءات ----------

    def stats(self) -> str:
        base = (f"rx={self.frames_received} proc={self.frames_processed} "
                f"drop={self.frames_dropped}")
        if _detector is not None and _detector.available:
            base += (f" backend={_detector.backend.name} "
                     f"infer={_detector.avg_ms:.1f}ms")
        return base
