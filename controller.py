"""
controller.py — المتحكم الاحترافي
==================================
AI-GP Virtual Qualifier | VADR-TS-002

آلة حالات:
    ARM → HOVER → RACE → (SEARCH | RECOVER) → RACE

حلقة تحكم متتالية:
    ملاحة (bearing / elevation / distance)
      → زوايا مستهدفة (roll / pitch) + معدل انعراج
        → معدلات دوران عبر PD
          → set_attitude_target  (وضع ACRO)

الفرق الجوهري عن النسخة السابقة:
    الرؤية موصولة فعلياً بالتحكم. سابقاً كان vision_rx يكشف
    البوابات ويخزّنها، والمتحكم يتجاهلها ويطير للأمام أعمى.
"""

from __future__ import annotations

import math
import time

from pymavlink import mavutil

from constants import CONTROL_HZ
from navigator import Navigator, NavTarget

# ─────────────────────────────────────────────────────────────
#  ثوابت
# ─────────────────────────────────────────────────────────────

MAVLINK_CMD_SIM_RESET = 31000
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

# --- الدفع والارتفاع ---
THRUST_BASE = 0.50
THRUST_MIN = 0.30
THRUST_MAX = 0.72
ALT_KP = 0.030
ALT_KI = 0.004
ALT_I_LIMIT = 0.08
CLIMB_KP = 0.35

# --- حلقة الزوايا (PD) ---
ANGLE_KP = 1.10
ANGLE_KD = 0.38
MAX_RATE = 2.2

# --- التوجيه ---
YAW_KP = 1.30
YAW_MAX = 1.8
ROLL_KP = 1.05
MAX_ROLL = 0.50
MAX_PITCH = 0.40
YAW_SLEW_MAX = 0.10   # حد تغيّر معدل الانعراج المطلوب لكل تِك — يمنع قفزة مفاجئة

# --- جدول السرعة ---
SPEED_MIN = 2.0
SPEED_CRUISE = 6.0
SPEED_MAX = 6.0
SPEED_KP = 0.075
SPEED_KD = 0.030
BRAKE_DIST = 4.0
ALIGN_PENALTY = 2.2

# سقف أمان السرعة: أعلى من هذا = الطائرة فعلياً خارج نطاق التحكم
# (اصطدام قذفها، أو تدحرجت وسقطت) — نقطع الدفع للحد الأدنى ونسوّي
# الوضعية بدل الاستمرار بالملاحة العادية فوق سرعة غير طبيعية.
SPEED_GOVERNOR = SPEED_MAX * 1.6

# --- التوقيت ---
TAKEOFF_SEC = 1.0
TAKEOFF_THRUST = 0.74
START_DELAY_SEC = 0.6
LOST_TIMEOUT_S = 1.5
RECOVER_SEC = 1.6
POST_RECOVER_RAMP_SEC = 2.0   # بعد الاسترداد: نعيد بناء السرعة تدريجياً لا فوراً
SLEW_MAX = 0.05


def _clamp(v, lo, hi=None):
    if hi is None:
        lo, hi = -lo, lo
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
#  إرسال الأوامر
# ─────────────────────────────────────────────────────────────

def send_attitude_rates(conn, boot_ms, roll_rate, pitch_rate, yaw_rate, thrust):
    conn.mav.set_attitude_target_send(
        int(time.time() * 1000) - boot_ms,
        conn.target_system,
        conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],
        float(roll_rate),
        float(pitch_rate),
        float(yaw_rate),
        float(thrust),
    )


def send_motors(conn, fl=0.5, fr=0.5, bl=0.5, br=0.5):
    conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        conn.target_system,
        conn.target_component,
        0,
        [fl, fr, bl, br, 0, 0, 0, 0],
    )


# ─────────────────────────────────────────────────────────────
#  المتحكم
# ─────────────────────────────────────────────────────────────

class Controller:

    def __init__(self, sim_conn, data, system_boot_ms):
        self.conn = sim_conn
        self.data = data
        self.boot_ms = system_boot_ms
        self.nav = Navigator(data)

        self.armed_at: float | None = None
        self._go_at: float | None = None
        self.state = "BOOT"

        self._alt_i = 0.0
        self._prev_roll_cmd = 0.0
        self._prev_pitch_cmd = 0.0
        self._prev_yaw_rate = 0.0
        self._last_target_t = 0.0
        self._recover_until = 0.0
        self._recovered_at: float | None = None
        self._was_recovering = False
        self._last_dbg = 0.0
        self._tick = 0
        self._collisions = 0

    # ---------- حالة ----------

    def _goto(self, state: str):
        if state != self.state:
            print(f"[STATE] {self.state} -> {state}  (t={self._elapsed():.1f}s)",
                  flush=True)
            self.state = state

    def _elapsed(self) -> float:
        """الوقت منذ إشارة بدء السباق الفعلية — لا منذ التسليح.

        الإقلاع نفسه يبدأ بهذا التوقيت (انظر update)، فلا نقلع مبكراً
        وننتظر بالجو — الإقلاع المبكر يُحتسب "بداية مبكرة" أيضاً.
        """
        return 0.0 if self._go_at is None else time.time() - self._go_at

    # ---------- ارتفاع ----------

    def _thrust(self, climb_target: float = 0.0) -> float:
        """NED: vel_z موجبة = هبوط. climb_target موجبة = نريد الصعود."""
        vel_z = self.data.get("vel_z", 0.0)
        err = climb_target + vel_z
        self._alt_i = _clamp(self._alt_i + err * ALT_KI, ALT_I_LIMIT)
        return _clamp(THRUST_BASE + ALT_KP * err + self._alt_i,
                      THRUST_MIN, THRUST_MAX)

    # ---------- سرعة ----------

    def _target_speed(self, nav: NavTarget) -> float:
        if not nav.has_target:
            return SPEED_MIN

        misalign = min(abs(nav.bearing) / (math.pi / 3), 1.0)
        f_align = 1.0 / (1.0 + ALIGN_PENALTY * misalign)

        f_near = min(nav.distance / BRAKE_DIST, 1.0)
        f_near = 0.45 + 0.55 * f_near

        ramp = min(self._elapsed() / 3.0, 1.0)
        # بعد استرداد من اصطدام: نعيد بناء السرعة تدريجياً، لا نقفز
        # فوراً لسرعة الإبحار — هذا كان يرمي الطائرة بسرعة داخل نفس
        # الخطأ الذي سبب الاصطدام الأول قبل ما تستقر وتعيد محاذاة نفسها.
        if self._recovered_at is not None:
            recover_ramp = min((time.time() - self._recovered_at) / POST_RECOVER_RAMP_SEC, 1.0)
            ramp = min(ramp, recover_ramp)

        return _clamp(SPEED_CRUISE * f_align * f_near * ramp,
                      SPEED_MIN * ramp, SPEED_MAX)

    # ---------- توجيه ----------

    def _guidance(self, nav: NavTarget):
        """ملاحة → (roll_cmd, pitch_cmd, yaw_rate, climb_target)"""
        yaw_rate = _clamp(YAW_KP * nav.bearing, YAW_MAX)
        yaw_rate = self._prev_yaw_rate + _clamp(yaw_rate - self._prev_yaw_rate, YAW_SLEW_MAX)
        self._prev_yaw_rate = yaw_rate
        roll_cmd = _clamp(ROLL_KP * nav.bearing, MAX_ROLL)

        vel_x = self.data.get("vel_x", 0.0)
        vel_y = self.data.get("vel_y", 0.0)
        speed = math.hypot(vel_x, vel_y)
        pitch_speed = self.data.get("pitch_speed", 0.0)

        v_err = self._target_speed(nav) - speed
        mag = abs(SPEED_KP * v_err - SPEED_KD * pitch_speed)
        # في NED الميل الأمامي زاوية سالبة
        pitch_cmd = _clamp(-mag if v_err > 0 else mag, MAX_PITCH)

        climb_target = _clamp(CLIMB_KP * nav.elevation * max(nav.distance, 1.0),
                              2.5)
        return roll_cmd, pitch_cmd, yaw_rate, climb_target

    # ---------- حلقة الزوايا ----------

    def _rates(self, roll_cmd: float, pitch_cmd: float):
        roll = self.data.get("roll", 0.0)
        pitch = self.data.get("pitch", 0.0)
        roll_sp = self.data.get("roll_speed", 0.0)
        pitch_sp = self.data.get("pitch_speed", 0.0)

        return (_clamp(ANGLE_KP * (roll_cmd - roll) - ANGLE_KD * roll_sp, MAX_RATE),
                _clamp(ANGLE_KP * (pitch_cmd - pitch) - ANGLE_KD * pitch_sp, MAX_RATE))

    def _slew(self, roll_cmd: float, pitch_cmd: float):
        roll_cmd = self._prev_roll_cmd + _clamp(roll_cmd - self._prev_roll_cmd, SLEW_MAX)
        pitch_cmd = self._prev_pitch_cmd + _clamp(pitch_cmd - self._prev_pitch_cmd, SLEW_MAX)
        self._prev_roll_cmd, self._prev_pitch_cmd = roll_cmd, pitch_cmd
        return roll_cmd, pitch_cmd

    # ---------- الحلقة الرئيسية ----------

    def update(self):
        self._tick += 1
        dt = 1.0 / CONTROL_HZ

        if self.armed_at is None:
            time.sleep(dt)
            return

        # مراقبة الاصطدام
        c = self.data.get("collision_count", 0)
        if c > self._collisions:
            self._collisions = c
            self._recover_until = time.time() + RECOVER_SEC
            self._goto("RECOVER")

        # ═══ ننتظر إشارة بدء السباق الرسمية — ما نقلع قبلها ═══
        # الإقلاع نفسه (حتى لو ما تحركنا نحو بوابة) يُحتسب "بداية مبكرة"
        # عند المحاكي. سابقاً كنا نقلع ونحوم بمؤقت محلي ثابت بعد التسليح
        # مباشرة، فنكون بالجو ومرتفعين لحظة ما يبدأ السباق فعلياً — وهذا
        # بحد ذاته عقوبة. الآن نبقى على الأرض (دفع صفري) لين تجي الإشارة.
        if not self.data.get("race_started"):
            self._goto("READY")
            self._go_at = None
            send_attitude_rates(self.conn, self.boot_ms, 0, 0, 0, 0.0)
            self._debug(0.0, None)
            time.sleep(dt)
            return

        if self._go_at is None:
            self._go_at = time.time()
        t = self._elapsed()

        # ═══ الإقلاع ═══
        if t < TAKEOFF_SEC:
            self._goto("ARM")
            send_motors(self.conn, *([TAKEOFF_THRUST] * 4))
            self._debug(TAKEOFF_THRUST, None)
            time.sleep(dt)
            return

        # ═══ تحويم للاستقرار ═══
        if t < TAKEOFF_SEC + START_DELAY_SEC:
            self._goto("HOVER")
            thr = self._thrust()
            send_attitude_rates(self.conn, self.boot_ms, 0, 0, 0, thr)
            self._debug(thr, None)
            time.sleep(dt)
            return

        # ═══ حاكم أمان السرعة ═══
        # لو السرعة تجاوزت حداً غير منطقي فعلياً (طيران حر بعد اصطدام
        # قذفها، أو انقلاب) — نقطع كل ملاحة فوراً ونخفّض الدفع للحد
        # الأدنى مع تسوية الوضعية، بدل الاستمرار بحلقة توجيه تفترض
        # طيراناً طبيعياً. هذا ما كان يحصل في الفيديو: بعد أول اصطدام
        # قرب بوابة رقم 1 استمرت الطائرة تتسارع (~65 كم/س) في الفراغ
        # بلا أي كابح.
        speed_now = math.hypot(self.data.get("vel_x", 0.0), self.data.get("vel_y", 0.0))
        if speed_now > SPEED_GOVERNOR:
            self._goto("GOVERN")
            rr, pr = self._rates(0.0, 0.0)
            send_attitude_rates(self.conn, self.boot_ms, rr, pr, 0.0, THRUST_MIN)
            self._debug(THRUST_MIN, None)
            time.sleep(dt)
            return

        # ═══ الملاحة ═══
        nav = self.nav.solve()
        if nav.has_target:
            self._last_target_t = time.time()

        # ═══ استرداد بعد اصطدام ═══
        if time.time() < self._recover_until:
            self._goto("RECOVER")
            self._was_recovering = True
            thr = self._thrust(climb_target=0.4)
            # نستخدم حلقة الزوايا لا معدلات صفرية مباشرة: في وضع ACRO
            # الأخيرة "تجمّد" أي ميلان تسبب فيه الاصطدام بدل تصحيحه،
            # فتستمر الطائرة بالتسارع بزاويتها المائلة بدل العودة للاستواء.
            rr, pr = self._rates(0.0, 0.0)
            send_attitude_rates(self.conn, self.boot_ms, rr, pr, 0.0, thr)
            self._debug(thr, nav)
            time.sleep(dt)
            return

        if self._was_recovering:
            self._was_recovering = False
            self._recovered_at = time.time()

        # ═══ هدف مفقود ═══
        if not nav.has_target:
            if time.time() - self._last_target_t > LOST_TIMEOUT_S:
                self._goto("SEARCH")
                thr = self._thrust()
                send_attitude_rates(self.conn, self.boot_ms, 0.0, -0.05, 0.45, thr)
                self._debug(thr, nav)
            else:
                rr, pr = self._rates(self._prev_roll_cmd, self._prev_pitch_cmd)
                send_attitude_rates(self.conn, self.boot_ms, rr, pr, 0.0,
                                    self._thrust())
            time.sleep(dt)
            return

        # ═══ السباق ═══
        self._goto("RACE")

        roll_cmd, pitch_cmd, yaw_rate, climb = self._guidance(nav)
        roll_cmd, pitch_cmd = self._slew(roll_cmd, pitch_cmd)
        roll_rate, pitch_rate = self._rates(roll_cmd, pitch_cmd)
        thrust = self._thrust(climb)

        send_attitude_rates(self.conn, self.boot_ms,
                            roll_rate, pitch_rate, yaw_rate, thrust)
        self._debug(thrust, nav)
        time.sleep(dt)

    # ---------- تشخيص ----------

    def _debug(self, thrust: float, nav):
        now = time.time()
        if now - self._last_dbg < 0.5:
            return
        self._last_dbg = now

        vx = self.data.get("vel_x", 0.0)
        vy = self.data.get("vel_y", 0.0)
        pz = self.data.get("pos_z", 0.0)
        spd = math.hypot(vx, vy)

        if nav and nav.has_target:
            print(f"[{self.state:7s}] alt={-pz:5.1f} spd={spd:4.1f} "
                  f"thr={thrust:.2f} | gate#{nav.gate_index} d={nav.distance:5.1f} "
                  f"brg={math.degrees(nav.bearing):+6.1f}° "
                  f"elv={math.degrees(nav.elevation):+5.1f}° src={nav.source}",
                  flush=True)
        else:
            print(f"[{self.state:7s}] alt={-pz:5.1f} spd={spd:4.1f} "
                  f"thr={thrust:.2f} | no target", flush=True)

    # ---------- أوامر ----------

    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self.armed_at = time.time()
        self._last_target_t = time.time()

    def send_sim_reset_command(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0,
        )
