"""
timesync.py — مزامنة الوقت مع المحاكي
=======================================

🐞 إصلاح علة: النسخة السابقة كانت تبدأ الخيط في classmethod
   اسمها create_timesync، لكن setup.py ينادي المُنشئ مباشرة
   TimeSync(sim_conn, shared_data) — فالخيط لا يبدأ أبداً
   ومزامنة الوقت لا تعمل. الآن المُنشئ يبدأ الخيط بنفسه.
"""

import threading
import time


TIMESYNC_REQUEST_HZ = 10


class TimeSync:

    def __init__(self, mavlink_connection, data, autostart: bool = True):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False
        if autostart:
            self.start()

    def start(self):
        if self.thread is not None:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self.timesync_loop, daemon=False)
        self.thread.start()

    @classmethod
    def create_timesync(cls, mavlink_connection, data):
        return cls(mavlink_connection, data, autostart=True)

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def timesync_loop(self):
        period = 1.0 / TIMESYNC_REQUEST_HZ
        while self.is_running:
            try:
                self.mavlink_conn.mav.timesync_send(int(time.time_ns()), 0)
            except Exception as e:
                print(f"[TIMESYNC] send failed: {e}", flush=True)
            time.sleep(period)
