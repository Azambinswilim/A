"""
shape_color_detector.py — كاشف احتياطي بالألوان/الأشكال للبوابات البرتقالية
=============================================================================
AI-GP Virtual Qualifier | VADR-TS-002

ليه نحتاجه؟
    نموذج YOLO (best.onnx / best_3_.pt) مدرَّب أساساً على بوابات مربعة الإطار،
    فهو يتعرف عليها بثقة عالية. أما بوابة الحلقة الدائرية (ring) البرتقالية،
    فحدّتها ومحيطها مختلفان بما يكفي إن الثقة تطلع تحت CONF_THRESHOLD ويتم
    رفضها بالكامل — فتختفي البوابة من نظر الطائرة تماماً رغم إنها موجودة
    وواضحة في الإطار.

    هذا الكاشف لا يعتمد على تعلّم آلي: يقصّ اللون البرتقالي (HSV) ثم يصنّف
    الشكل الناتج (مربع مقابل حلقة) عبر محيط الكفاف وعدد أضلاعه ونسبة
    الاستدارة. يُستخدم لتعويض ما يفوت النموذج، لا لاستبداله — يُدمج مع
    مخرجات YOLO في detector.py.

لماذا نشترط وجود "ثقب" (hole)؟
    أي بوابة سباق فعلية جسم مجوّف (الطائرة تعبر من المنتصف)، فتظهر في
    القناع الثنائي ككفاف خارجي له كفاف داخلي (ثقب) عبر RETR_CCOMP.
    اشتراط الثقب يستبعد أي جسم برتقالي صلب (مخروط، لافتة) قد يخدع كاشف
    لون بسيط بدونه.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from constants import (
    CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT,
    GATE_COLOR_HSV_LOW_1, GATE_COLOR_HSV_HIGH_1,
    GATE_COLOR_HSV_LOW_2, GATE_COLOR_HSV_HIGH_2,
    GATE_COLOR_MIN_AREA_PX,
)

# دائرة "مثالية" لها circularity = 1.0 مقابل π/4≈0.785 للمربع — قريبتان
# جداً لتمييزهما بمفردهما، فعدد أضلاع approxPolyDP هو الفيصل (مربع ~4،
# حلقة ~8+) والاستدارة تأكيد إضافي فقط.
RING_CIRCULARITY_MIN = 0.83
SQUARE_MAX_VERTICES = 5              # حواف الإطار المربع قد "تتكسر" لأكثر من 4
SQUARE_MAX_ASPECT = 1.6              # نسبة طول/عرض أقصى لاعتباره مربعاً لا مستطيلاً عشوائياً

_MORPH_KERNEL = np.ones((5, 5), np.uint8)


def _orange_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(GATE_COLOR_HSV_LOW_1), np.array(GATE_COLOR_HSV_HIGH_1))
    m2 = cv2.inRange(hsv, np.array(GATE_COLOR_HSV_LOW_2), np.array(GATE_COLOR_HSV_HIGH_2))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=1)
    return mask


def _classify_shape(contour, outer_area: float) -> tuple[str, float]:
    """يرجّع (shape, quality) حيث quality في [0, 1] يعكس نظافة تطابق الشكل."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-3:
        return "unknown", 0.0

    circularity = 4.0 * math.pi * outer_area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1.0, min(w, h))

    # المربع أولاً: عدد أضلاعه القليل حاسم ولا يلتبس بدائرة حتى لو
    # اقتربت استدارته من π/4 (~0.785) بسبب سماكة الإطار.
    if len(approx) <= SQUARE_MAX_VERTICES and aspect <= SQUARE_MAX_ASPECT \
            and cv2.isContourConvex(cv2.convexHull(approx)):
        quality = 1.0 - min(1.0, abs(aspect - 1.0))
        return "square", max(0.4, quality)

    if circularity >= RING_CIRCULARITY_MIN:
        quality = min(1.0, circularity)
        return "ring", quality

    return "unknown", 0.0


def detect_orange_gates(img_bgr: np.ndarray) -> list[dict]:
    """
    يكتشف بوابات (مربعة أو حلقية) برتقالية اللون بمعالجة صورة كلاسيكية.
    يرجّع نفس قاموس detector._pack زائداً "shape" و "color_score".
    """
    mask = _orange_mask(img_bgr)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    out = []
    for i, contour in enumerate(contours):
        parent = hierarchy[i][3]
        if parent != -1:
            continue   # هذا كفاف ثقب داخلي، لا كفاف بوابة خارجي

        outer_area = cv2.contourArea(contour)
        if outer_area < GATE_COLOR_MIN_AREA_PX:
            continue

        child = hierarchy[i][2]
        if child == -1:
            continue   # لا ثقب = على الأرجح جسم برتقالي صلب، ليست بوابة

        shape, quality = _classify_shape(contour, outer_area)
        if shape == "unknown":
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

        # ثقة اصطناعية: هذا كاشف كلاسيكي لا نموذج مدرَّب، فالثقة تعكس
        # نظافة تطابق الشكل فقط — تُبقى دون سقف نموذج YOLO الجيد عمداً
        # حتى لا "تتنافس" بلا داعٍ مع كشف YOLO الموثوق حين يتفقان.
        confidence = min(0.85, 0.5 + 0.35 * quality)

        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": w, "h": h,
            "area": (w * h) / (CAMERA_IMAGE_WIDTH * CAMERA_IMAGE_HEIGHT),
            "confidence": float(confidence),
            "shape": shape,
            "color_score": 1.0,
            "source": "color",
        })

    return out
