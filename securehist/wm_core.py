"""Blind fragile DWT-SVD watermark (vectorised).

Embedding: every 16x16 block of the Cb channel -> 1-level Haar DWT -> SVD of the
horizontal (cH) and vertical (cV) detail sub-bands -> QIM of the K largest singular
values onto a lattice selected by a per-block key bit + key dither.
Verification needs only the key: no original or reference image.

Key derivation
  secret == b""  : seed = SHA-256(f"{patient_id}_{row}_{col}")   (legacy, reproduces
                   model_dwt_svd_blind.py; NOT secure - anyone who knows the patient
                   ID can re-embed a valid watermark)
  secret != b""  : seed = HMAC-SHA-256(secret, f"{patient_id}_{row}_{col}")  (use this)
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_opening, label, uniform_filter


@dataclass(frozen=True)
class WMConfig:
    block: int = 16
    step: float = 8.0          # QIM step on singular values
    k: int = 2                 # singular values quantised per sub-band
    passes: int = 8            # re-embedding passes so bits survive uint8 rounding
    tamper_threshold: float = 0.19
    smoothing: int = 3         # k x k mean filter on the block distance map
    opening: bool = True       # 2x2 morphological opening of the flagged mask
    min_region_blocks: int = 8 # flagged regions smaller than this are discarded (8 blocks = 2048 px)
    # threshold / region size fixed on a development subset of 80 images (10 per class folder):
    # 0/80 untouched images flagged, 96% of 64x64 copy-move edits flagged
    nc_intact: float = 0.50    # image-level: NC >= this -> watermark present/intact
    domain: str = "dwt_svd"    # embedding domain: "dwt_svd" (ours), "dwt_coef" or "dft_mag" (ablation only)


# Fixed carrier positions for the ablation domains (4 carriers per block, like 2 sigma x 2 bands)
DWT_COEF_POS = ((1, 2), (2, 1))                    # in each 8x8 detail band (cH, cV)
DFT_POS = ((2, 5), (5, 2), (3, 4), (4, 3))          # mid-frequency (u, v) in the 16x16 block DFT


def secret_from_env(var: str = "SECUREHIST_WM_SECRET") -> bytes:
    """Secret key from an environment variable (empty -> legacy, insecure mode)."""
    return os.environ.get(var, "").encode("utf-8")


# ----------------------------------------------------------------------------- keys
_KEY_CACHE: dict = {}


def block_keys(patient_id: str, nby: int, nbx: int, n_bits: int, secret: bytes = b""):
    """Per-block key bits (N, n_bits) in {0,1} and dither (N, n_bits) in [0,1)."""
    ck = (patient_id, nby, nbx, n_bits, secret)
    if ck in _KEY_CACHE:
        return _KEY_CACHE[ck]
    bits = np.empty((nby * nbx, n_bits), dtype=np.int64)
    dith = np.empty((nby * nbx, n_bits), dtype=np.float64)
    idx = 0
    for i in range(nby):
        for j in range(nbx):
            msg = f"{patient_id}_{i}_{j}".encode("utf-8")
            if secret:
                digest = hmac.new(secret, msg, hashlib.sha256).hexdigest()
            else:
                digest = hashlib.sha256(msg).hexdigest()
            rng = np.random.default_rng(int(digest[:8], 16))
            bits[idx] = rng.integers(0, 2, n_bits)
            dith[idx] = rng.uniform(0.0, 1.0, n_bits)
            idx += 1
    _KEY_CACHE[ck] = (bits, dith)
    return bits, dith


# ------------------------------------------------------------------- block helpers
def crop_to_blocks(img: np.ndarray, block: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % block, : w - w % block]


def to_blocks(ch: np.ndarray, b: int) -> np.ndarray:
    h, w = ch.shape
    return ch.reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b, b)


def from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    b = blocks.shape[-1]
    return blocks.reshape(h // b, w // b, b, b).transpose(0, 2, 1, 3).reshape(h, w)


def haar_dwt(x: np.ndarray):
    """1-level orthonormal Haar on a stack (N, b, b) -> LL, cH, cV, cD each (N, b/2, b/2)."""
    a, b_ = x[:, 0::2, 0::2], x[:, 0::2, 1::2]
    c, d = x[:, 1::2, 0::2], x[:, 1::2, 1::2]
    return (a + b_ + c + d) / 2, (a + b_ - c - d) / 2, (a - b_ + c - d) / 2, (a - b_ - c + d) / 2


def haar_idwt(LL, cH, cV, cD):
    n, m, _ = LL.shape
    out = np.empty((n, 2 * m, 2 * m), dtype=np.float64)
    out[:, 0::2, 0::2] = (LL + cH + cV + cD) / 2
    out[:, 0::2, 1::2] = (LL + cH - cV - cD) / 2
    out[:, 1::2, 0::2] = (LL - cH + cV - cD) / 2
    out[:, 1::2, 1::2] = (LL - cH - cV + cD) / 2
    return out


def _qim_offsets(bits, dith, step):
    return (dith + 0.5 * bits) * step


# ------------------------------------------------------- ablation carriers (not ours)
def _carriers(blocks: np.ndarray, cfg: WMConfig) -> np.ndarray:
    """Carrier values (N, 4) for the ablation domains."""
    if cfg.domain == "dwt_coef":
        _, cH, cV, _ = haar_dwt(blocks)
        return np.stack([band[:, r, c] for band in (cH, cV) for (r, c) in DWT_COEF_POS], axis=1)
    if cfg.domain == "dft_mag":
        F = np.fft.fft2(blocks, norm="ortho")
        return np.stack([np.abs(F[:, u, v]) for (u, v) in DFT_POS], axis=1)
    raise ValueError(cfg.domain)


def _set_carriers(blocks: np.ndarray, values: np.ndarray, cfg: WMConfig) -> np.ndarray:
    if cfg.domain == "dwt_coef":
        LL, cH, cV, cD = haar_dwt(blocks)
        i = 0
        for band in (cH, cV):
            for (r, c) in DWT_COEF_POS:
                band[:, r, c] = values[:, i]; i += 1
        return haar_idwt(LL, cH, cV, cD)
    if cfg.domain == "dft_mag":
        F = np.fft.fft2(blocks, norm="ortho")
        b = blocks.shape[-1]
        for i, (u, v) in enumerate(DFT_POS):
            mag = np.maximum(values[:, i], 0.0)
            ph = np.angle(F[:, u, v])
            F[:, u, v] = mag * np.exp(1j * ph)
            F[:, (-u) % b, (-v) % b] = mag * np.exp(-1j * ph)       # keep the block real
        return np.real(np.fft.ifft2(F, norm="ortho"))
    raise ValueError(cfg.domain)


# ------------------------------------------------------------------------ embedding
def _embed_pass(rgb: np.ndarray, patient_id: str, cfg: WMConfig, secret: bytes) -> np.ndarray:
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    cb = ycc[:, :, 2]
    h, w = cb.shape
    blocks = to_blocks(cb, cfg.block)
    bits, dith = block_keys(patient_id, h // cfg.block, w // cfg.block, 2 * cfg.k, secret)
    if cfg.domain != "dwt_svd":
        vals = _carriers(blocks, cfg)
        off = _qim_offsets(bits, dith, cfg.step)
        vals = np.round((vals - off) / cfg.step) * cfg.step + off
        if cfg.domain == "dft_mag":
            vals = np.where(vals < 0, vals + cfg.step, vals)       # magnitudes cannot be negative
        ycc[:, :, 2] = from_blocks(_set_carriers(blocks, vals, cfg), h, w)
        return cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    LL, cH, cV, cD = haar_dwt(blocks)
    new = []
    for band_idx, band in enumerate((cH, cV)):
        U, S, Vt = np.linalg.svd(band)
        sl = slice(band_idx * cfg.k, (band_idx + 1) * cfg.k)
        off = _qim_offsets(bits[:, sl], dith[:, sl], cfg.step)
        S[:, : cfg.k] = np.round((S[:, : cfg.k] - off) / cfg.step) * cfg.step + off
        new.append((U * S[:, None, :]) @ Vt)
    ycc[:, :, 2] = from_blocks(haar_idwt(LL, new[0], new[1], cD), h, w)
    return cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def embed(rgb: np.ndarray, patient_id: str, cfg: WMConfig = WMConfig(), secret: bytes = b"") -> np.ndarray:
    """Watermark an RGB uint8 image (cropped to a multiple of the block size)."""
    out = crop_to_blocks(rgb, cfg.block)
    for _ in range(cfg.passes):
        out = _embed_pass(out, patient_id, cfg, secret)
    return out


# --------------------------------------------------------------------- verification
def block_distances(rgb: np.ndarray, patient_id: str, cfg: WMConfig = WMConfig(), secret: bytes = b""):
    """Per-block mean QIM distance (0 intact .. ~0.25 random) and per-bit correctness.

    Returns dist_map (nby, nbx) and bit_ok (N, 2k) boolean.
    """
    rgb = crop_to_blocks(rgb, cfg.block)
    cb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)[:, :, 2]
    h, w = cb.shape
    blocks = to_blocks(cb, cfg.block)
    bits, dith = block_keys(patient_id, h // cfg.block, w // cfg.block, 2 * cfg.k, secret)
    if cfg.domain != "dwt_svd":
        r = (_carriers(blocks, cfg) - _qim_offsets(bits, dith, cfg.step)) / cfg.step
        d = np.abs(r - np.round(r))
        return d.mean(axis=1).reshape(h // cfg.block, w // cfg.block), d < 0.25
    LL, cH, cV, cD = haar_dwt(blocks)
    d = []
    for band_idx, band in enumerate((cH, cV)):
        S = np.linalg.svd(band, compute_uv=False)[:, : cfg.k]
        sl = slice(band_idx * cfg.k, (band_idx + 1) * cfg.k)
        r = (S - _qim_offsets(bits[:, sl], dith[:, sl], cfg.step)) / cfg.step
        d.append(np.abs(r - np.round(r)))
    d = np.concatenate(d, axis=1)
    return d.mean(axis=1).reshape(h // cfg.block, w // cfg.block), d < 0.25


def nc_score(rgb: np.ndarray, patient_id: str, cfg: WMConfig = WMConfig(), secret: bytes = b"") -> float:
    """Image-level blind NC = 1 - 2*BER over all key bits (1 intact, ~0 absent/wrong key)."""
    _, ok = block_distances(rgb, patient_id, cfg, secret)
    return float(2.0 * ok.mean() - 1.0)


def tamper_map(rgb: np.ndarray, patient_id: str, cfg: WMConfig = WMConfig(), secret: bytes = b""):
    """Smoothed block score map and boolean flagged-block mask (True = tampered)."""
    dist, _ = block_distances(rgb, patient_id, cfg, secret)
    score = uniform_filter(dist, cfg.smoothing, mode="nearest") if cfg.smoothing > 1 else dist
    flagged = score > cfg.tamper_threshold
    if cfg.opening:
        flagged = binary_opening(flagged, structure=np.ones((2, 2), dtype=bool))
    if cfg.min_region_blocks > 1 and flagged.any():
        lab, n = label(flagged)
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes[1:] >= cfg.min_region_blocks
        flagged = keep[lab]
    return score, flagged


def verify(rgb: np.ndarray, patient_id: str, cfg: WMConfig = WMConfig(), secret: bytes = b"") -> dict:
    """All blind signals for one image."""
    score, flagged = tamper_map(rgb, patient_id, cfg, secret)
    dist, ok = block_distances(rgb, patient_id, cfg, secret)
    nc = float(2.0 * ok.mean() - 1.0)
    return {
        "nc": nc,
        "flag_frac": float(flagged.mean()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
        "intact": bool(nc >= cfg.nc_intact and not flagged.any()),
        "score_map": score,
        "flagged": flagged,
    }


# ---------------------------------------------------------- adaptive attack helper
def restore_singular_values(attacked_rgb: np.ndarray, reference_rgb: np.ndarray, cfg: WMConfig = WMConfig()) -> np.ndarray:
    """Key-less adaptive step: copy sigma_1..sigma_k of cH/cV from the watermarked image
    into the attacked image block-wise (the attacker can see the watermarked image but
    not the key). Used by the adaptive 'SVD-preserving' attack in eval_system.py."""
    a = cv2.cvtColor(crop_to_blocks(attacked_rgb, cfg.block), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    r = cv2.cvtColor(crop_to_blocks(reference_rgb, cfg.block), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    h, w = a.shape[:2]
    LLa, cHa, cVa, cDa = haar_dwt(to_blocks(a[:, :, 2], cfg.block))
    _, cHr, cVr, _ = haar_dwt(to_blocks(r[:, :, 2], cfg.block))
    new = []
    for ba, br in ((cHa, cHr), (cVa, cVr)):
        U, S, Vt = np.linalg.svd(ba)
        Sr = np.linalg.svd(br, compute_uv=False)
        S[:, : cfg.k] = Sr[:, : cfg.k]
        new.append((U * S[:, None, :]) @ Vt)
    a[:, :, 2] = from_blocks(haar_idwt(LLa, new[0], new[1], cDa), h, w)
    return cv2.cvtColor(np.clip(np.round(a), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def restore_carriers(attacked_rgb: np.ndarray, reference_rgb: np.ndarray, cfg: WMConfig = WMConfig()) -> np.ndarray:
    """Key-less adaptive step: copy carriers from the watermarked reference image into
    the attacked image block-wise (attacker knows the published carrier positions/domain, but not the key).
    For dwt_svd: copies sigma_1..sigma_k.
    For dwt_coef: copies DWT_COEF_POS detail coefficients.
    For dft_mag: copies DFT_POS magnitudes."""
    if cfg.domain == "dwt_svd":
        return restore_singular_values(attacked_rgb, reference_rgb, cfg)
    a = cv2.cvtColor(crop_to_blocks(attacked_rgb, cfg.block), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    r = cv2.cvtColor(crop_to_blocks(reference_rgb, cfg.block), cv2.COLOR_RGB2YCrCb).astype(np.float64)
    h, w = a.shape[:2]
    ba = to_blocks(a[:, :, 2], cfg.block)
    br = to_blocks(r[:, :, 2], cfg.block)
    vals_r = _carriers(br, cfg)
    ba_restored = _set_carriers(ba, vals_r, cfg)
    a[:, :, 2] = from_blocks(ba_restored, h, w)
    return cv2.cvtColor(np.clip(np.round(a), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def block_mask_from_pixels(mask: np.ndarray, block: int = 16) -> np.ndarray:
    """Pixel mask (H, W) -> block ground truth (nby, nbx): True if any pixel in block."""
    m = crop_to_blocks(mask.astype(bool), block)
    h, w = m.shape
    return m.reshape(h // block, block, w // block, block).any(axis=(1, 3))
