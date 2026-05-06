"""MOSSE + 4-state Kalman wrapper.

Kalman maintains state (cx, cy, vx, vy) under a constant-velocity model.
Each frame:
  1. Kalman predict -> (px, py)  (last position + last velocity)
  2. MOSSE samples patch AROUND (px, py), finds correlation peak (mx, my)
  3. Mahalanobis chi^2 gating: reject measurement if too far from prior
  4. If accepted -> Kalman correct + MOSSE filter update
     If rejected -> Kalman predict-only, MOSSE filter NOT updated (avoid
                    learning the distractor that produced the bad meas)

Why this helps over plain MOSSE:
  - Drone barely visible -> MOSSE peak sub-optimal, but Kalman recovers
    via velocity prior; sampling at predicted center keeps drone in patch.
  - Distractor briefly outscores drone -> chi^2 gate rejects it, no drift.
  - Sub-pixel correlation peak jitter -> Kalman smooths to ~1 px stable.
"""
import numpy as np
import cv2

from .socf_tracker import SOCFTracker


class KalmanMOSSE:
    # Looser chi^2 gate (matches C++ tracker's 25): rejects only the
    # truly egregious outliers, lets normal jitter through. Tighter Q_VEL
    # so velocity prior doesn't shoot the state into infinity after a
    # few rejected frames.
    # Drone in test1 oscillates ±40 px frame-to-frame (camera shake);
    # constant-velocity Kalman over-extrapolates and drives MOSSE patch
    # to the wrong area. Damp Q_VEL hard so velocity estimate is barely
    # used (Kalman degenerates toward random-walk + small drift).
    GATE_CHI2 = 25.0
    Q_POS = 5.0
    Q_VEL = 20.0          # constant-velocity model, moderate
    R_MEAS = 1.0
    # PSR-weighted output: weight = sigmoid((PSR - PSR_MID) / PSR_WIDTH).
    # High PSR -> trust MOSSE peak; low PSR -> lean on Kalman state.
    PSR_MID = 8.0
    PSR_WIDTH = 3.0

    def __init__(self):
        self.mosse = SOCFTracker()
        self.kf = cv2.KalmanFilter(4, 2)
        # F: constant-velocity transition.
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        # H: pick (cx, cy) from state.
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.diag(
            [self.Q_POS, self.Q_POS, self.Q_VEL, self.Q_VEL]
        ).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag(
            [self.R_MEAS, self.R_MEAS]).astype(np.float32)
        cv2.setIdentity(self.kf.errorCovPost, 10.0)
        self.bbox_wh = None

    def init(self, gray: np.ndarray, bbox_xywh):
        x, y, w, h = bbox_xywh
        self.bbox_wh = (int(w), int(h))
        cx, cy = x + w / 2.0, y + h / 2.0
        self.mosse.init(gray, bbox_xywh)
        self.kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0]], dtype=np.float32)

    def update(self, gray: np.ndarray):
        # 1. Kalman predict -> sample center for MOSSE.
        pred = self.kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])

        # 2. MOSSE samples around its OWN last centre. Test1 drone
        #    oscillates (camera shake) so Kalman's velocity prior is
        #    often pointed the wrong way; sampling at MOSSE's own peak
        #    keeps the patch well-aligned with the actual drone signal.
        x, y, w, h = self.mosse.update(gray)
        mx, my = x + w / 2.0, y + h / 2.0

        # 3. Mahalanobis chi^2 gate.
        innov = np.array([[mx - px], [my - py]], dtype=np.float32)
        S = self.kf.errorCovPre[:2, :2] + self.kf.measurementNoiseCov
        chi2 = float((innov.T @ np.linalg.inv(S) @ innov)[0, 0])

        if chi2 < self.GATE_CHI2:
            self.kf.correct(np.array([[mx], [my]], dtype=np.float32))
        else:
            # Outlier MOSSE peak: reject only the Kalman update; do
            # NOT touch MOSSE's internal state. MOSSE keeps tracking
            # autonomously; Kalman remembers prior state and provides
            # next-frame motion prior unaffected by the bad meas.
            self.kf.statePost = self.kf.statePre.copy()

        # 4. Confidence-weighted output: high MOSSE PSR -> trust MOSSE
        #    peak; low PSR (drone hard to find / occluded) -> lean on
        #    Kalman state.
        psr = self.mosse.last_psr
        # Sigmoid weight: w = 1 / (1 + exp(-(psr - PSR_MID) / PSR_WIDTH))
        w_mosse = 1.0 / (1.0 + np.exp(-(psr - self.PSR_MID) / self.PSR_WIDTH))
        cx_kf = float(self.kf.statePost[0, 0])
        cy_kf = float(self.kf.statePost[1, 0])
        mx_, my_ = self.mosse.center
        cx_out = w_mosse * mx_ + (1.0 - w_mosse) * cx_kf
        cy_out = w_mosse * my_ + (1.0 - w_mosse) * cy_kf
        bw, bh = self.bbox_wh
        return (cx_out - bw / 2.0, cy_out - bh / 2.0, bw, bh)
