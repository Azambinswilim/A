"""
detector.py — كاشف البوابات (يدعم ONNX و PyTorch)
===================================================
AI-GP Virtual Qualifier | VADR-TS-002

لماذا طبقة تجريد؟
    عندك صيغتان للنموذج نفسه:
      best_3_.pt   — PyTorch، أسرع مع GPU (CUDA)
      best.onnx    — ONNX، مستقل، أفضل على CPU-only و Jetson

    هذه الوحدة تخفي الفرق. تختار الأسرع تلقائياً، أو تجبرها
    عبر متغير البيئة GATE_BACKEND=onnx|torch

معالجة ONNX الخام:
    مخرج النموذج [1, 5, 8400] = (cx, cy, w, h, conf) لكل مرساة.
    نطبّق العتبة ثم NMS يدوياً — النموذج مُصدَّر بـ nms=False.

Letterbox:
    الكاميرا 640×360 والنموذج يريد 640×640. نضيف أشرطة رمادية
    فوق وتحت (لا نمطّ الصورة) ثم نعكس التحويل على الإحداثيات.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from constants import CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT
from shape_color_detector import detect_orange_gates, orange_mask

# ─────────────────────────────────────────────────────────────
#  الإعدادات
# ─────────────────────────────────────────────────────────────

# المسارات الافتراضية نسبةً لمجلد هذا الملف نفسه، لا لمجلد العمل
# الحالي (cwd). سابقاً كانت "best.onnx" / "best_3_.pt" نسبية لـ cwd —
# فلو شغّلت main.py من أي مجلد غير مجلد المشروع (مثلاً من IDE، أو عبر
# سكربت تشغيل خارجي)، os.path.exists() يفشل بصمت، الكاشف يرجع None،
# ويطبع فقط سطر تحذير سهل تفويته في الطرفية — وتتحول الملاحة بالكامل
# للهندسة فقط، أي أن "الـ AI" فعلياً لا يرى/يتعرف على أي حلقة أبداً.
_HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.getenv("GATE_ONNX", os.path.join(_HERE, "best.onnx"))
TORCH_PATH = os.getenv("GATE_PT", os.path.join(_HERE, "best_3_.pt"))
BACKEND = os.getenv("GATE_BACKEND", "auto").lower()   # auto | onnx | torch

CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 10

MODEL_SIZE = 640          # مقاس مدخل النموذج (مربع)
PAD_VALUE = 114           # قيمة الأشرطة الرمادية


# ─────────────────────────────────────────────────────────────
#  أدوات
# ─────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int = MODEL_SIZE):
    """
    يغيّر المقاس مع الحفاظ على النسبة ويضيف حشواً.
    يرجّع (صورة_مربعة, نسبة, إزاحة_x, إزاحة_y)
    """
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))

    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw, dh = size - nw, size - nh
    top, left = dh // 2, dw // 2
    out = cv2.copyMakeBorder(img, top, dh - top, left, dw - left,
                             cv2.BORDER_CONSTANT,
                             value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    return out, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """كبت غير الأعظمي — boxes بصيغة xyxy."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        order = order[1:][iou <= iou_thr]

    return keep[:MAX_DETECTIONS]


# ─────────────────────────────────────────────────────────────
#  الخلفيات
# ─────────────────────────────────────────────────────────────

class OnnxBackend:
    """استنتاج ONNX الخام — مستقل عن PyTorch."""

    name = "onnx"

    def __init__(self, path: str = ONNX_PATH):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        avail = ort.get_available_providers()
        for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider",
                  "CoreMLExecutionProvider", "CPUExecutionProvider"):
            if p in avail:
                providers.append(p)

        self.sess = ort.InferenceSession(path, sess_options=opts,
                                         providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]

        # إحماء
        blank = np.zeros((CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH, 3), np.uint8)
        self.infer(blank)

    def infer(self, img_bgr: np.ndarray) -> list[dict]:
        padded, ratio, dx, dy = letterbox(img_bgr)

        blob = padded[:, :, ::-1]                       # BGR -> RGB
        blob = blob.transpose(2, 0, 1)[None]            # HWC -> NCHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        out = self.sess.run(None, {self.input_name: blob})[0]

        # [1, 5, 8400] -> [8400, 5]
        pred = out[0].T
        scores = pred[:, 4]
        mask = scores > CONF_THRESHOLD
        if not mask.any():
            return []

        pred = pred[mask]
        scores = scores[mask]

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2,
                          cx + w / 2, cy + h / 2], axis=1)

        # عكس الـ letterbox للرجوع لإحداثيات الصورة الأصلية
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / ratio

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, CAMERA_IMAGE_WIDTH)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, CAMERA_IMAGE_HEIGHT)

        keep = nms(boxes, scores, IOU_THRESHOLD)
        return [_pack(boxes[i], float(scores[i]), img_bgr) for i in keep]


class TorchBackend:
    """استنتاج Ultralytics — أسرع مع CUDA."""

    name = "torch"

    def __init__(self, path: str = TORCH_PATH):
        from ultralytics import YOLO
        self.model = YOLO(path)

        try:
            import torch
            self.provider = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.provider = "cpu"

        blank = np.zeros((CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH, 3), np.uint8)
        self.infer(blank)

    def infer(self, img_bgr: np.ndarray) -> list[dict]:
        res = self.model.predict(img_bgr, conf=CONF_THRESHOLD,
                                 iou=IOU_THRESHOLD, verbose=False,
                                 imgsz=MODEL_SIZE)[0]
        out = []
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            out.append(_pack(np.array(xyxy), float(b.conf[0]), img_bgr))
        return out


def _color_score(img_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    """نسبة بكسلات الصندوق التي تقع ضمن مدى البرتقالي — يفرّق بين بوابة
    حقيقية وأي مربع/جسم آخر يشبهها بالشكل لكن ليس بلونها. يستخدم نفس
    قناع shape_color_detector (مع تسوية الإضاءة) فلا يفترض إضاءة ثابتة."""
    h_img, w_img = img_bgr.shape[:2]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w_img, int(x2)), min(h_img, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    crop = img_bgr[yi1:yi2, xi1:xi2]
    mask = orange_mask(crop)
    return float(np.count_nonzero(mask)) / float(mask.size)


def _pack(xyxy, conf: float, img_bgr: np.ndarray | None = None) -> dict:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = x2 - x1, y2 - y1
    d = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "w": w, "h": h,
        "area": (w * h) / (CAMERA_IMAGE_WIDTH * CAMERA_IMAGE_HEIGHT),
        "confidence": conf,
        "shape": "unknown",
        "source": "model",
    }
    d["color_score"] = _color_score(img_bgr, x1, y1, x2, y2) if img_bgr is not None else 0.0
    return d


def _iou_xyxy(a: dict, b: dict) -> float:
    xx1, yy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    xx2, yy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter + 1e-9)


def _merge_detections(model_dets: list[dict], color_dets: list[dict]) -> list[dict]:
    """يدمج كشف النموذج مع الكشف الكلاسيكي بالألوان/الأشكال.

    النموذج يتعرف على المربعات بثقة عالية لكنه يفوّت الحلقات الدائرية
    غالباً (راجع رأس shape_color_detector.py). فحين يتفق الاثنان على نفس
    الصندوق تقريباً نبقي كشف النموذج (أدق) وننسخ له تصنيف الشكل إن كان
    مجهولاً؛ وحين لا يوجد تطابق (حالة الحلقة التي فاتت النموذج) نضيف كشف
    الألوان كمرشّح مستقل بدل تجاهله."""
    merged = list(model_dets)
    for c in color_dets:
        best_iou = 0.0
        best_idx = -1
        for i, m in enumerate(merged):
            iou = _iou_xyxy(c, m)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou > 0.3:
            if merged[best_idx].get("shape") == "unknown":
                merged[best_idx]["shape"] = c["shape"]
            merged[best_idx]["color_score"] = max(merged[best_idx].get("color_score", 0.0), c["color_score"])
        else:
            merged.append(c)
    return merged[:MAX_DETECTIONS]


# ─────────────────────────────────────────────────────────────
#  الواجهة
# ─────────────────────────────────────────────────────────────

class GateDetector:
    """
    كاشف موحّد. الاستخدام:
        det = GateDetector()
        boxes = det(frame_bgr)
    """

    def __init__(self):
        self.backend = self._select()
        self.infer_ms = 0.0
        self._ema_ms = None
        self.frames = 0

        if self.backend is None:
            print("[DETECT] no backend available — geometry-only navigation",
                  flush=True)
        else:
            print(f"[DETECT] backend={self.backend.name} "
                  f"device={self.backend.provider}", flush=True)

    def _select(self):
        order = []
        if BACKEND == "onnx":
            order = [(OnnxBackend, ONNX_PATH)]
        elif BACKEND == "torch":
            order = [(TorchBackend, TORCH_PATH)]
        else:
            # auto: torch أسرع بس مع CUDA فعلياً. من غيرها overhead
            # الـ Python بتاع ultralytics يخليه أبطأ بكثير وغير ثابت
            # (شفنا فرق حتى 20x مقابل onnx على CPU) — لدرجة إنه ممكن
            # يجوّع خيط MAVLink ويجمّد التيليمتري. لو مافيه CUDA، onnx
            # أولاً أضمن. onnx نفسه يجرّب CUDA/TensorRT قبل CPU لو متاحة.
            cuda = False
            try:
                import torch
                cuda = torch.cuda.is_available()
            except Exception:
                cuda = False

            if cuda:
                order = [(TorchBackend, TORCH_PATH), (OnnxBackend, ONNX_PATH)]
            else:
                order = [(OnnxBackend, ONNX_PATH), (TorchBackend, TORCH_PATH)]

        missing = []
        for cls, path in order:
            if not os.path.exists(path):
                missing.append(path)
                continue
            try:
                t0 = time.perf_counter()
                b = cls(path)
                print(f"[DETECT] loaded {path} "
                      f"in {(time.perf_counter()-t0):.1f}s", flush=True)
                return b
            except Exception as e:
                print(f"[DETECT] {cls.name} unavailable: {e}", flush=True)

        if missing:
            print("[DETECT] !! model file(s) not found, tried: "
                  + ", ".join(missing), flush=True)
            print("[DETECT] !! vision is DISABLED — navigation will run on "
                  "track geometry only. Set GATE_ONNX / GATE_PT env vars "
                  "if the weights live elsewhere.", flush=True)
        return None

    def __call__(self, img_bgr: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()

        model_dets = []
        if self.backend is not None:
            try:
                model_dets = self.backend.infer(img_bgr)
            except Exception as e:
                print(f"[DETECT] inference failed: {e}", flush=True)
                model_dets = []

        # الكاشف الكلاسيكي (لون + شكل) يعمل دائماً — لا يعتمد على النموذج،
        # ويعوّض بوابات الحلقة الدائرية التي يفوّتها النموذج غالباً.
        try:
            color_dets = detect_orange_gates(img_bgr)
        except Exception as e:
            print(f"[DETECT] color/shape fallback failed: {e}", flush=True)
            color_dets = []

        dets = _merge_detections(model_dets, color_dets)

        ms = (time.perf_counter() - t0) * 1000.0
        self.infer_ms = ms
        self._ema_ms = ms if self._ema_ms is None else 0.9 * self._ema_ms + 0.1 * ms
        self.frames += 1
        return dets

    @property
    def avg_ms(self) -> float:
        return self._ema_ms or 0.0

    @property
    def available(self) -> bool:
        # الكاشف الكلاسيكي بالألوان يعمل دائماً حتى بدون وزن النموذج،
        # فالرؤية تبقى "متاحة" وإن كان النموذج مفقوداً.
        return True


# ─────────────────────────────────────────────────────────────
#  اختبار سريع:  python detector.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    det = GateDetector()
    if not det.available:
        sys.exit("no backend")

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            sys.exit(f"cannot read {sys.argv[1]}")
    else:
        img = np.random.randint(0, 255,
                                (CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH, 3),
                                dtype=np.uint8)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        boxes = det(img)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    med = times[len(times) // 2]
    print(f"\nmedian {med:.1f} ms  ->  {1000/med:.1f} FPS "
          f"(camera runs at 30 FPS, budget 33 ms)")
    print(f"detections: {len(boxes)}")
    for b in boxes:
        print(f"   conf={b['confidence']:.2f} "
              f"center=({b['cx']:.0f},{b['cy']:.0f}) area={b['area']:.4f}")
