"""Spectral Residual Saliency (Hou & Zhang, CVPR 2007).

Frequency-domain saliency: subtract local mean of log-amplitude spectrum,
inverse FFT, square magnitude, blur. Highlights compact "anomalous"
intensity patches against textured / repetitive background.

Used by SOCF (2023) as the object-aware mask multiplied into the
correlation filter so the filter learns to focus on the saliency-
peaked region (i.e. the drone, even when low-contrast).
"""
import numpy as np
import cv2


def spectral_residual(gray_patch: np.ndarray) -> np.ndarray:
    """Return saliency map (float32, normalised to [0, 1]) of input gray patch.

    Args:
        gray_patch: HxW uint8 or float gray image.
    Returns:
        HxW float32 in [0, 1]; high = salient.
    """
    img = gray_patch.astype(np.float32)
    F = np.fft.fft2(img)
    log_amp = np.log(np.abs(F) + 1e-9)
    phase = np.angle(F)
    # Local average of log-spectrum (3x3 box) - "natural" smooth spectrum
    log_amp_avg = cv2.boxFilter(log_amp, -1, (3, 3))
    # Spectral residual = deviation from natural spectrum
    spectral_resid = log_amp - log_amp_avg
    sal_complex = np.fft.ifft2(np.exp(spectral_resid + 1j * phase))
    sal = np.abs(sal_complex) ** 2
    sal = cv2.GaussianBlur(sal, (5, 5), sigmaX=8.0)
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max - sal_min < 1e-9:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - sal_min) / (sal_max - sal_min)).astype(np.float32)
