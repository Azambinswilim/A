# ----------------------------------------------------------------------------
# AI-GP Virtual Qualifier — Official Competition Constants
# Source: VADR-TS-002 Technical Specification (Issue 00.02)
# ----------------------------------------------------------------------------

# --- Camera / Vision Stream (Section 3.8 + 4.6) ---
CAMERA_IMAGE_WIDTH  = 640       # px
CAMERA_IMAGE_HEIGHT = 360       # px
CAMERA_FPS          = 30        # Hz  (vision stream frequency)
CAMERA_TILT_DEG     = 20.0      # degrees, camera tilted UP from body frame
CAMERA_VFOV_DEG     = 90.0      # degrees, vertical field of view

# Pinhole camera intrinsics (no lens distortion)
CAMERA_CX, CAMERA_CY = 320.0, 180.0   # principal point [px]
CAMERA_FX, CAMERA_FY = 320.0, 320.0   # focal length [px]

# --- Control Loop (Section 4.4) ---
# Spec caps command rate at <100Hz. Team-approved rate: 60Hz.
CONTROL_HZ = 60

# --- Networking (Section 4.2 / 4.6) ---
MAVLINK_UDP_PORT = 14550
VISION_STREAM_UDP_PORT = 5600

# --- Physics reference (Section 3.2, informational) ---
PHYSICS_HZ = 120

# --- Gate color (Section 3.4: gates are painted safety-orange, both the
#     square-frame gates and the circular ring/hoop gates) ---
# HSV bounds (OpenCV H: 0-179, S/V: 0-255). Two bands because "orange" in
# HSV can sit close to the red wrap-around depending on lighting. Kept
# deliberately loose on S/V — different worlds/levels light the gates
# differently, and we normalize brightness (CLAHE) before thresholding
# anyway, so hue is the only value that has to be trustworthy.
GATE_COLOR_HSV_LOW_1  = (5, 80, 50)
GATE_COLOR_HSV_HIGH_1 = (22, 255, 255)
GATE_COLOR_HSV_LOW_2  = (0, 80, 50)
GATE_COLOR_HSV_HIGH_2 = (4, 255, 255)

# Minimum blob area (in px, at native 640x360) for the classical
# color/shape fallback to consider a contour a candidate gate.
GATE_COLOR_MIN_AREA_PX = 150
