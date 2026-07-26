"""
navigator.py — الملاحة: دمج هندسة المسار مع التصحيح البصري
============================================================
AI-GP Virtual Qualifier | VADR-TS-002

الفكرة الأساسية:
    المحاكي يرسل لنا إحداثيات كل البوابات (NED) + مؤشر البوابة الهدف.
    هذا يعطينا اتجاهاً هندسياً دقيقاً — لا يضيع ولا يرمش.
    الرؤية (YOLO) تصحّح انحراف تقدير الموقع، وتؤكد أننا على الهدف.

    هندسة  = الاتجاه العام (موثوق، متاح دائماً)
    رؤية   = تصحيح دقيق قرب البوابة (دقيق، متقطّع)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from constants import (
    CAMERA_CX, CAMERA_CY, CAMERA_FX, CAMERA_FY,
    CAMERA_TILT_DEG, CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT,
)

# ─────────────────────────────────────────────────────────────
#  الإعدادات
# ─────────────────────────────────────────────────────────────

VISION_MAX_AGE_S = 0.25     # أقدم من كذا = كشف بائت، لا نثق فيه
VISION_BLEND_NEAR = 0.65    # وزن الرؤية عند القرب من البوابة
VISION_BLEND_FAR = 0.20     # وزن الرؤية عند البعد
BLEND_NEAR_DIST = 6.0       # [m] تحتها نعتبر أنفسنا قريبين

# نقطة الهدف = مركز البوابة + (المتجه الأمامي × LOOKAHEAD_M).
# كانت 1.0م سابقاً: قريبة جداً من مستوى البوابة، فكلما اقتربت الطائرة
# منها كانت المسافة إلى الهدف تؤول لقيمة صغيرة و atan2(dy,dx) يكبّر أي
# انحراف جانبي بسيط إلى زاوية ميلان/انعراج ضخمة في آخر لحظة (شوهد في
# الفيديو: الطائرة تنزلق نحو حافة البوابة وتصطدم بالإطار بدل المرور
# من المنتصف). تكبير المسافة يجعل خط الاقتراب أنعم ويمنع هذا الانفجار.
LOOKAHEAD_M = 2.2

# تنعيم أسّي (EMA) على bearing/elevation المدمجة — يمتص أي قفزة كشف
# منفردة (رؤية أو هندسة) بدل تمريرها مباشرة لحلقة التحكم كأمر مفاجئ.
NAV_SMOOTH_ALPHA = 0.55     # 0 = بلا تنعيم، أقرب لـ1 = تنعيم أثقل

CAM_TILT_RAD = math.radians(CAMERA_TILT_DEG)


# ─────────────────────────────────────────────────────────────
#  أنواع البيانات
# ─────────────────────────────────────────────────────────────

@dataclass
class Gate:
    """بوابة سباق بإحداثيات NED."""
    gate_id: int
    x: float          # North [m]
    y: float          # East  [m]
    z: float          # Down  [m]  (سالب = فوق سطح الأرض)
    qw: float
    qx: float
    qy: float
    qz: float
    width: float      # [m]
    height: float     # [m]

    def normal(self) -> tuple[float, float, float]:
        """متجه الاتجاه الأمامي للبوابة (محور X المحلي مدوّراً)."""
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        return (1 - 2 * (y * y + z * z),
                2 * (x * y + w * z),
                2 * (x * z - w * y))

    def approach_point(self, ahead: float = LOOKAHEAD_M) -> tuple[float, float, float]:
        """نقطة خلف البوابة على محورها — نستهدفها بدل مركزها بالضبط."""
        nx, ny, nz = self.normal()
        return (self.x + nx * ahead, self.y + ny * ahead, self.z + nz * ahead)


@dataclass
class NavTarget:
    """مخرجات الملاحة — ما يحتاجه المتحكم."""
    bearing: float = 0.0        # [rad] خطأ الاتجاه الأفقي (+ = الهدف يميناً)
    elevation: float = 0.0      # [rad] خطأ الاتجاه الرأسي (+ = الهدف فوق)
    distance: float = 99.0      # [m]  المسافة للهدف
    has_target: bool = False
    source: str = "none"        # "fused" | "geometry" | "vision" | "none"
    gate_index: int = 0
    confidence: float = 0.0


# ─────────────────────────────────────────────────────────────
#  الملاح
# ─────────────────────────────────────────────────────────────

class Navigator:

    def __init__(self, data: dict):
        self.data = data
        self._last_idx = -1
        self._gate_log: list[tuple[int, float]] = []
        self._smooth_bearing: float | None = None
        self._smooth_elevation: float | None = None

    # ---------- مصادر ----------

    def _gates(self) -> list[Gate]:
        return self.data.get("track_gates") or []

    def _target_gate(self) -> Gate | None:
        gates = self._gates()
        if not gates:
            return None
        idx = int(self.data.get("active_gate_index", 0))
        if idx != self._last_idx:
            if self._last_idx >= 0:
                self._gate_log.append((self._last_idx, time.time()))
                print(f"[NAV] gate {self._last_idx} cleared -> now targeting {idx}",
                      flush=True)
            self._last_idx = idx
            # بوابة جديدة = اتجاه جديد كلياً، لا نريد للتنعيم أن يجرّنا
            # نحو اتجاه البوابة القديمة لحظة التبديل.
            self._smooth_bearing = None
            self._smooth_elevation = None
        if 0 <= idx < len(gates):
            return gates[idx]
        return gates[-1]

    # ---------- هندسة ----------

    def _geometric(self) -> NavTarget | None:
        """اتجاه الهدف من إحداثيات NED — الأساس الموثوق."""
        gate = self._target_gate()
        if gate is None:
            return None

        px = self.data.get("pos_x")
        py = self.data.get("pos_y")
        pz = self.data.get("pos_z")
        if px is None or py is None or pz is None:
            return None

        tx, ty, tz = gate.approach_point()
        dx, dy, dz = tx - px, ty - py, tz - pz

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-3:
            return None

        yaw = self.data.get("yaw", 0.0)

        # الاتجاه المطلوب مقابل اتجاهنا الحالي
        desired_yaw = math.atan2(dy, dx)
        bearing = _wrap_pi(desired_yaw - yaw)

        # الارتفاع: NED فـ dz سالب يعني الهدف أعلى منا
        horiz = math.hypot(dx, dy)
        elevation = math.atan2(-dz, max(horiz, 1e-3))

        return NavTarget(
            bearing=bearing,
            elevation=elevation,
            distance=dist,
            has_target=True,
            source="geometry",
            gate_index=self._last_idx,
            confidence=1.0,
        )

    # ---------- رؤية ----------

    def _visual(self) -> NavTarget | None:
        """اتجاه الهدف من صندوق YOLO — دقيق لكن متقطّع."""
        det = self.data.get("gate_detection")
        if not det:
            return None

        stamp = self.data.get("gate_detection_time", 0.0)
        if time.time() - stamp > VISION_MAX_AGE_S:
            return None      # كشف بائت

        cx, cy = det["cx"], det["cy"]
        w = max(det["x2"] - det["x1"], 1.0)

        # زوايا من نموذج الثقب الإبري
        bearing = math.atan2(cx - CAMERA_CX, CAMERA_FX)
        elev_cam = math.atan2(CAMERA_CY - cy, CAMERA_FY)

        # الكاميرا مائلة +20° لأعلى — نضيف الميلان للحصول على الزاوية بإطار الجسم
        elevation = elev_cam + CAM_TILT_RAD

        # المسافة من العرض الظاهر
        gate = self._target_gate()
        real_w = gate.width if gate else 1.5
        dist = (real_w * CAMERA_FX) / w

        return NavTarget(
            bearing=bearing,
            elevation=elevation,
            distance=float(min(max(dist, 0.3), 100.0)),
            has_target=True,
            source="vision",
            gate_index=self._last_idx,
            confidence=float(det.get("confidence", 0.0)),
        )

    # ---------- الدمج ----------

    def solve(self) -> NavTarget:
        geo = self._geometric()
        vis = self._visual()

        if geo is None and vis is None:
            self._smooth_bearing = None
            self._smooth_elevation = None
            return NavTarget()

        if vis is None:
            return self._smoothed(geo)

        if geo is None:
            return self._smoothed(vis)

        # وزن الرؤية يزيد كلما اقتربنا (الصندوق يصير أدق)
        t = min(geo.distance / BLEND_NEAR_DIST, 1.0)
        w_vis = VISION_BLEND_NEAR + (VISION_BLEND_FAR - VISION_BLEND_NEAR) * t
        w_vis *= vis.confidence      # ثقة منخفضة = وزن أقل

        # لو الرؤية تختلف جذرياً عن الهندسة، فهي على الأرجح بوابة أخرى
        if abs(_wrap_pi(vis.bearing - geo.bearing)) > math.radians(35):
            w_vis *= 0.25

        w_geo = 1.0 - w_vis

        fused = NavTarget(
            bearing=w_geo * geo.bearing + w_vis * vis.bearing,
            elevation=w_geo * geo.elevation + w_vis * vis.elevation,
            distance=min(geo.distance, vis.distance),
            has_target=True,
            source="fused",
            gate_index=geo.gate_index,
            confidence=vis.confidence,
        )
        return self._smoothed(fused)

    def _smoothed(self, tgt: NavTarget) -> NavTarget:
        """تنعيم أسّي لـ bearing/elevation فقط — يمنع قفزة كشف واحدة
        (ضجيج رؤية أو تذبذب هندسي قرب الهدف) من التحوّل لأمر ميلان/
        انعراج مفاجئ. يُعاد ضبطه فور تبديل البوابة الهدف (_target_gate)."""
        a = NAV_SMOOTH_ALPHA
        if self._smooth_bearing is None:
            self._smooth_bearing = tgt.bearing
            self._smooth_elevation = tgt.elevation
        else:
            db = _wrap_pi(tgt.bearing - self._smooth_bearing)
            self._smooth_bearing = _wrap_pi(self._smooth_bearing + (1 - a) * db)
            self._smooth_elevation += (1 - a) * (tgt.elevation - self._smooth_elevation)

        tgt.bearing = self._smooth_bearing
        tgt.elevation = self._smooth_elevation
        return tgt


# ─────────────────────────────────────────────────────────────
#  أدوات
# ─────────────────────────────────────────────────────────────

def _wrap_pi(a: float) -> float:
    """يلفّ زاوية إلى المدى [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a
