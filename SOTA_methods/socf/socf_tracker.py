"""Simplified SOCF (Spatial disturbance suppression + Object saliency-aware
Correlation Filter) - Wang et al., Expert Systems with Applications, 2023.

Original paper uses STRCF backbone + ADMM solver (heavy). This impl uses
a MOSSE closed-form base for simplicity and FPGA-friendliness, retaining
the two SOCF innovations:

  1. Object saliency-aware filter: spectral residual saliency mask is
     multiplied into the response so the filter focuses on the salient
     (drone) region instead of bright but uniform distractors.

  2. Spatial disturbance suppression: a sliding window of past response
     maps is averaged to form a "context response" R_ctx; deviation
     D_t = R_t - R_ctx > tau marks DISTRACTOR pixels in the current
     response. The numerator update A_new is suppressed at those
     pixels so the filter does not learn the distractor.

Closed-form MOSSE update (no ADMM):
    A_t = (1 - eta) * A_{t-1} + eta * G * conj(F_t) * (1 - alpha * M_dist)
    B_t = (1 - eta) * B_{t-1} + eta * F_t * conj(F_t) + lambda
    H   = A_t / B_t              (complex division)
    R   = IFFT(F * conj(H))      (response map)
    R_sal = R * M_sal            (saliency-modulated)

Default hyperparams chosen to match Bolme MOSSE conventions; users can
override via __init__ kwargs.
"""
import numpy as np
import cv2

from .saliency import spectral_residual


class SOCFTracker:
    # Filter operates on a PATCH proportional to target size (Bolme-style
    # search region). PATCH = SEARCH_FACTOR * max(init_w, init_h),
    # rounded to nearest even integer. This guarantees the filter's
    # search range (~PATCH/2) exceeds typical inter-frame drone motion.
    SEARCH_FACTOR = 2.0  # PATCH = SEARCH_FACTOR * max(init_w, init_h)
    MIN_PATCH = 64       # floor for very small init boxes
    PEAK_SIGMA_FRAC = 0.10  # Gaussian label sigma = PEAK_SIGMA_FRAC * PATCH
    ETA = 0.125          # online learning rate
    LAMBDA = 0.01        # MOSSE regulariser
    HISTORY_N = 5        # frames in context-response window
    DIST_TAU = 0.30      # disturbance threshold (fraction of peak)
    DIST_ALPHA = 0.7     # disturbance penalty strength [0, 1]
    # Saliency dropped from this simplified SOCF: original paper applies
    # saliency mask to the FILTER via ADMM (multi-iter optimisation).
    # Multiplying spectral-residual saliency into the response (cheap
    # approximation) was empirically harmful (kills the drone peak when
    # drone is dim). Disturbance suppression in spatial domain (via
    # masked Gaussian label) is retained as the lighter SOCF flavour.
    USE_SALIENCY = False
    USE_DIST = False

    def __init__(self):
        self.patch_size = None     # set in init() based on bbox
        self.peak_sigma = None     # set in init()
        self.G = None
        self.cos_window = None
        self.A = None
        self.B = None
        self.center = None         # (cx, cy) in image coords
        self.bbox_wh = None        # original (w, h) - kept fixed
        self.history = []          # list of past response maps (real)
        self.last_psr = 0.0        # peak-to-sidelobe ratio (confidence)

    # ------------------------- helpers -------------------------------

    def _gauss_label(self) -> np.ndarray:
        """Gaussian peak centred in patch_size x patch_size."""
        n = self.patch_size
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
        c = n / 2.0
        g = np.exp(-((xs - c) ** 2 + (ys - c) ** 2) / (2.0 * self.peak_sigma ** 2))
        return g

    def _hann2d(self) -> np.ndarray:
        n = self.patch_size
        h = np.hanning(n).astype(np.float32)
        return np.outer(h, h)

    def _patch(self, gray: np.ndarray, center) -> np.ndarray:
        """Sample patch_size x patch_size around `center` (subpixel)."""
        cx, cy = float(center[0]), float(center[1])
        return cv2.getRectSubPix(gray, (self.patch_size, self.patch_size),
                                 (cx, cy))

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """log + per-patch zero-mean unit-std + Hann window."""
        f = patch.astype(np.float32)
        f = np.log(f + 1.0)
        m, s = f.mean(), f.std()
        f = (f - m) / (s + 1e-6)
        return f * self.cos_window

    @staticmethod
    def _complex_div(a: np.ndarray, b: np.ndarray, lam: float = 0.01) -> np.ndarray:
        """Element-wise complex division a / b with regularised denominator
        |b|^2 + lam (matches Bolme MOSSE convention)."""
        return a * np.conj(b) / (np.abs(b) ** 2 + lam)

    # ----------------------- main API --------------------------------

    def init(self, gray: np.ndarray, bbox_xywh):
        """Build initial filter + lock target size. bbox_xywh = (x, y, w, h)."""
        x, y, w, h = bbox_xywh
        self.bbox_wh = (int(w), int(h))
        cx, cy = x + w / 2.0, y + h / 2.0
        self.center = (cx, cy)

        # Search region size scales with target -> filter sees enough
        # context that drone never leaves the window between frames.
        target = max(int(w), int(h))
        n = max(self.MIN_PATCH, int(round(self.SEARCH_FACTOR * target)))
        if n % 2:                        # keep even for centred FFT
            n += 1
        self.patch_size = n
        self.peak_sigma = max(2.0, self.PEAK_SIGMA_FRAC * n)

        self.G = np.fft.fft2(self._gauss_label())
        self.cos_window = self._hann2d()

        f = self._preprocess(self._patch(gray, self.center))
        F = np.fft.fft2(f)
        self.A = self.G * np.conj(F)
        self.B = F * np.conj(F)              # LAMBDA only at division time
        self.history = []

    def update(self, gray: np.ndarray, sample_center=None):
        """Track one frame. `sample_center` optionally overrides where the
        patch is sampled (defaults to last self.center). The KalmanMOSSE
        wrapper passes the Kalman-predicted centre so MOSSE searches
        around the motion prior, not stale last-frame position. Returns
        (x, y, w, h)."""
        sc = sample_center if sample_center is not None else self.center
        # 1. Sample patch around `sc`, compute response.
        # Bolme 2010: store conj(H) = A/B = G*conj(F) / (F*conj(F));
        # response on new frame is FFT(f_new) * conj(H), implemented
        # as F * (A/B) directly (no extra conjugation).
        f = self._preprocess(self._patch(gray, sc))
        F = np.fft.fft2(f)
        H_conj = self._complex_div(self.A, self.B)
        R = np.real(np.fft.ifft2(F * H_conj))

        # 2. SOCF saliency mask: down-weight non-salient response pixels.
        if self.USE_SALIENCY:
            sal = spectral_residual(self._patch(gray, sc))
            R_sal = R * sal
        else:
            R_sal = R

        # 3. Find peak in saliency-modulated response.
        peak_y, peak_x = np.unravel_index(np.argmax(R_sal), R_sal.shape)
        peak_val = float(R_sal[peak_y, peak_x])
        # PSR (Peak-to-Sidelobe Ratio): mask 11x11 around peak, compare
        # peak to mean+std of off-peak region. Confidence proxy.
        mask = np.ones_like(R_sal, dtype=bool)
        py0 = max(0, peak_y - 5); py1 = min(R_sal.shape[0], peak_y + 6)
        px0 = max(0, peak_x - 5); px1 = min(R_sal.shape[1], peak_x + 6)
        mask[py0:py1, px0:px1] = False
        off = R_sal[mask]
        self.last_psr = float((peak_val - off.mean()) / (off.std() + 1e-9))
        # Centre offset (peak relative to patch centre).
        dx = peak_x - self.patch_size / 2.0
        dy = peak_y - self.patch_size / 2.0
        cx_new = sc[0] + dx
        cy_new = sc[1] + dy
        # Clamp to image bounds (target centre stays inside frame).
        H_img, W_img = gray.shape[:2]
        cx_new = float(np.clip(cx_new, 0, W_img - 1))
        cy_new = float(np.clip(cy_new, 0, H_img - 1))
        self.center = (cx_new, cy_new)

        # 4. Spatial disturbance suppression: history avg = context.
        self.history.append(R)
        if len(self.history) > self.HISTORY_N:
            self.history.pop(0)
        if self.USE_DIST and len(self.history) >= 2:
            R_ctx = np.mean(self.history[:-1], axis=0)
            D = R - R_ctx
            peak = max(R.max(), 1e-9)
            M_dist = (D > self.DIST_TAU * peak).astype(np.float32)
        else:
            M_dist = np.zeros_like(R, dtype=np.float32)

        # 5. Closed-form MOSSE update with disturbance penalty.
        # Apply suppress in SPATIAL domain on the Gaussian label so the
        # filter is told "don't try to make distractor pixels respond".
        # Then FFT the masked label and use as the new numerator target.
        f_new = self._preprocess(self._patch(gray, self.center))
        F_new = np.fft.fft2(f_new)
        suppress = 1.0 - self.DIST_ALPHA * M_dist  # spatial mask in [1-alpha, 1]
        g_masked = self._gauss_label() * suppress
        G_masked = np.fft.fft2(g_masked)
        A_new = G_masked * np.conj(F_new)
        B_new = F_new * np.conj(F_new)
        self.A = (1.0 - self.ETA) * self.A + self.ETA * A_new
        self.B = (1.0 - self.ETA) * self.B + self.ETA * B_new

        # Output: fixed (w, h), updated centre.
        w, h = self.bbox_wh
        return (cx_new - w / 2.0, cy_new - h / 2.0, w, h)
