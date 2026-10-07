#!/usr/bin/env python3
"""
Educational lossless JPEG2000-style pipeline (grayscale) with a figure per step.

  1. DC level shift
  2. Tiling
  3. Multi-level 2D reversible 5/3 DWT
  4. Coefficient statistics (sparsity)
  5. Bit-planes (what the Tier-1 coder works on)
  6. Entropy-coding cost (ESTIMATE per code-block; not the real MQ coder)
  7. Exact reconstruction (inverse DWT)
  8. Resolution scalability and bit-plane truncation

Usage:
  uv run j2k_steps.py [image] [--tile 128] [--levels 3] [--cblk 32]
                      [--out out] [--opj-bin DIR] [--no-show]
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =====================================================================
# 1D reversible 5/3 wavelet (integer lifting, whole-sample symmetric
# extension, even length)
# =====================================================================
def _nxt(a):  # a[i+1], mirrored at the end
    return np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)


def _prv(a):  # a[i-1], mirrored at the start
    return np.concatenate([a[..., :1], a[..., :-1]], axis=-1)


def _interleave(e, o):
    out = np.empty(e.shape[:-1] + (e.shape[-1] + o.shape[-1],), dtype=e.dtype)
    out[..., 0::2], out[..., 1::2] = e, o
    return out


def fwd53(x):
    """Returns (low, high)."""
    x = x.astype(np.int64)
    e, o = x[..., 0::2], x[..., 1::2]
    d = o - ((e + _nxt(e)) >> 1)             # high-pass (predict)
    s = e + ((_prv(d) + d + 2) >> 2)         # low-pass (update)
    return s, d


def inv53(s, d):
    e = s - ((_prv(d) + d + 2) >> 2)
    o = d + ((e + _nxt(e)) >> 1)
    return _interleave(e, o)


# =====================================================================
# 2D multi-level DWT stored as a "mosaic" (LL top-left, details around it).
# Vertical filtering first, then horizontal (as in the standard).
# =====================================================================
def fdwt2(tile, levels):
    out = tile.astype(np.int64).copy()
    h, w = tile.shape
    for _ in range(levels):
        L, H = fwd53(out[:h, :w].T)
        L, H = L.T, H.T                      # vertical low / high
        LL, HL = fwd53(L)                    # horizontal on vertical-low
        LH, HH = fwd53(H)                    # horizontal on vertical-high
        out[:h // 2, :w // 2], out[:h // 2, w // 2:h] = LL, HL
        out[h // 2:h, :w // 2], out[h // 2:h, w // 2:w] = LH, HH
        h, w = h // 2, w // 2
    return out


def idwt2(coef, levels):
    out = coef.copy()
    h, w = coef.shape[0] >> levels, coef.shape[1] >> levels
    for _ in range(levels):
        h, w = h * 2, w * 2
        LL, HL = out[:h // 2, :w // 2], out[:h // 2, w // 2:w]
        LH, HH = out[h // 2:h, :w // 2], out[h // 2:h, w // 2:w]
        L, H = inv53(LL, HL), inv53(LH, HH)
        out[:h, :w] = inv53(L.T, H.T).T
    return out


def subbands(T, levels):
    """(name, row_slice, col_slice) of every subband inside a TxT mosaic."""
    res = []
    for l in range(1, levels + 1):
        h = T >> l
        res += [(f"HL{l}", slice(0, h), slice(h, 2 * h)),
                (f"LH{l}", slice(h, 2 * h), slice(0, h)),
                (f"HH{l}", slice(h, 2 * h), slice(h, 2 * h))]
    h = T >> levels
    res.append((f"LL{levels}", slice(0, h), slice(0, h)))
    return res


# =====================================================================
# Entropy-cost estimate
# =====================================================================
def est_bits(q):
    """
    Rough Tier-1-like cost of one code-block, in bits. Goes bit-plane by
    bit-plane; each bit is conditioned on a context (already significant, or
    0/1/2/3+ significant neighbours); cost is the static entropy per context
    plus 1 bit per sign. NOT the real EBCOT/MQ coder, just a proxy.
    """
    mag = np.abs(q)
    m = int(mag.max())
    if m == 0:
        return 0.0
    sig = np.zeros(q.shape, bool)
    h, w = q.shape
    total = 0.0
    for p in range(m.bit_length() - 1, -1, -1):
        bit = ((mag >> p) & 1).astype(bool)
        pad = np.pad(sig, 1).astype(np.int8)
        nbr = sum(pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
        ctx = np.where(sig, 4, np.minimum(nbr, 3))
        for c in range(5):
            sel = ctx == c
            n = int(sel.sum())
            if n == 0:
                continue
            pr = bit[sel].sum() / n
            if 0 < pr < 1:
                total += -n * (pr * np.log2(pr) + (1 - pr) * np.log2(1 - pr))
        total += (bit & ~sig).sum()          # sign bits of newly significant
        sig |= bit
    return float(total)


# =====================================================================
# Helpers
# =====================================================================
def logvis(c):
    return np.sign(c) * np.log1p(np.abs(c))


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def synthetic(n=512):
    yy, xx = np.mgrid[0:n, 0:n]
    img = 128 + 60 * np.sin(xx / 40) * np.cos(yy / 55)
    img[((xx - 180) ** 2 + (yy - 200) ** 2) < 80 ** 2] = 220
    img[300:420, 280:460] = 40
    rng = np.random.default_rng(0)
    img[330:400, 300:440] += rng.normal(0, 25, (70, 140))
    return np.clip(img, 0, 255).astype(np.uint8)


def savefig(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=130)


def try_openjpeg(img, a, est_bytes):
    exe = shutil.which("opj_compress", path=a.opj_bin or None)
    if not exe:
        print("[OpenJPEG] opj_compress not found (use --opj-bin DIR to compare real size).")
        return
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.pgm"), os.path.join(td, "out.j2k")
        Image.fromarray(img).save(src)
        cmd = [exe, "-i", src, "-o", dst, "-t", f"{a.tile},{a.tile}",
               "-n", str(a.levels + 1), "-b", f"{a.cblk},{a.cblk}"]
        subprocess.run(cmd, check=True, capture_output=True)
        size = os.path.getsize(dst)
    print(f"[OpenJPEG] real codestream: {size} bytes | raw: {img.size} bytes | "
          f"ratio {img.size / size:.2f}:1 | my estimate: {est_bytes:.0f} bytes")


# =====================================================================
# Main pipeline
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--cblk", type=int, default=32)
    ap.add_argument("--out", default="jpeg2000/out")
    ap.add_argument("--opj-bin", default=None)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    T, L = a.tile, a.levels
    if T % (2 ** L):
        raise SystemExit("--tile must be divisible by 2**levels")
    os.makedirs(a.out, exist_ok=True)

    # ---- load + crop to a whole number of tiles ----
    img = np.array(Image.open(a.image).convert("L")) if a.image else synthetic()
    H, W = (img.shape[0] // T) * T, (img.shape[1] // T) * T
    if H == 0 or W == 0:
        raise SystemExit("image is smaller than one tile")
    img = img[:H, :W]
    print(f"Image {W}x{H}, lossless 5/3, tile={T}, levels={L}, cblk={a.cblk}")

    # ---- 1. DC level shift ----
    shifted = img.astype(np.int64) - 128
    print(f"[1] level shift: [{img.min()},{img.max()}] -> [{shifted.min()},{shifted.max()}]")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original (0..255)")
    im = ax[1].imshow(shifted, cmap="gray", vmin=-128, vmax=127)
    ax[1].set_title("Level-shifted (-128..127)"); plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].hist(img.ravel(), 64, alpha=.6, label="before")
    ax[2].hist(shifted.ravel(), 64, alpha=.6, label="after"); ax[2].legend()
    ax[2].set_title("Histogram: same shape, centred on 0")
    for x in ax[:2]: x.axis("off")
    savefig(fig, a.out, "01_level_shift.png")

    # ---- 2. Tiling ----
    tiles = [(y, x) for y in range(0, H, T) for x in range(0, W, T)]
    print(f"[2] tiling: {len(tiles)} tiles of {T}x{T}")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for v in range(0, W + 1, T): ax.axvline(v - .5, color="red", lw=1)
    for v in range(0, H + 1, T): ax.axhline(v - .5, color="red", lw=1)
    for i, (y, x) in enumerate(tiles):
        ax.text(x + T / 2, y + T / 2, str(i), color="yellow", ha="center", va="center")
    ax.set_title(f"{len(tiles)} tiles ({T}x{T})"); ax.axis("off")
    savefig(fig, a.out, "02_tiling.png")

    # ---- 3. DWT ----
    q, full = {}, np.zeros((H, W), np.int64)     # q: integer coefficients per tile
    for (y, x) in tiles:
        c = fdwt2(shifted[y:y + T, x:x + T], L)
        q[(y, x)] = c
        full[y:y + T, x:x + T] = c
    ctr = tiles[len(tiles) // 2]
    print(f"[3] DWT done. Coefficient range [{full.min()},{full.max()}]")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    ax[0].imshow(logvis(full), cmap="gray"); ax[0].set_title("DWT mosaic of every tile (log scale)")
    ax[1].imshow(logvis(q[ctr]), cmap="gray"); ax[1].set_title(f"Tile at (x,y)={ctr[::-1]}, subbands")
    for name, rs, cs in subbands(T, L):
        ax[1].add_patch(plt.Rectangle((cs.start - .5, rs.start - .5), cs.stop - cs.start,
                                      rs.stop - rs.start, fill=False, ec="red", lw=.8))
        ax[1].text((cs.start + cs.stop) / 2, (rs.start + rs.stop) / 2, name,
                   color="yellow", ha="center", va="center", fontsize=8)
    for x in ax: x.axis("off")
    savefig(fig, a.out, "03_dwt.png")

    # ---- 4. Coefficient statistics ----
    z0, z1 = np.mean(shifted == 0), np.mean(full == 0)
    print(f"[4] zeros: {z0:.1%} of samples before DWT -> {z1:.1%} of coefficients after")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(shifted.ravel(), 256, alpha=.6, label="pixels (level-shifted)")
    ax.hist(full.ravel(), 256, alpha=.6, label="DWT coefficients")
    ax.set_yscale("log"); ax.legend(); ax.set_title("Histogram (log y): DWT concentrates values near 0")
    savefig(fig, a.out, "04_coefficient_histogram.png")

    # ---- 5. Bit-planes of the centre tile ----
    mag = np.abs(q[ctr]); nb = max(int(mag.max()).bit_length(), 1)
    planes = list(range(nb - 1, max(nb - 9, -1), -1))
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax_, p in zip(axs.ravel(), planes):
        ax_.imshow((mag >> p) & 1, cmap="gray", vmin=0, vmax=1); ax_.set_title(f"bit-plane {p}"); ax_.axis("off")
    for ax_ in axs.ravel()[len(planes):]: ax_.axis("off")
    fig.suptitle(f"Magnitude bit-planes of tile {ctr[::-1]} (MSB first): what Tier-1 codes")
    savefig(fig, a.out, "05_bitplanes.png")
    print(f"[5] bit-planes: centre tile has {nb} magnitude bit-planes")

    # ---- 6. Entropy coding cost estimate per code-block ----
    heat, total = np.zeros((H, W)), 0.0
    for (y, x), qt in q.items():
        for name, rs, cs in subbands(T, L):
            cb = min(a.cblk, rs.stop - rs.start)
            for by in range(rs.start, rs.stop, cb):
                for bx in range(cs.start, cs.stop, cb):
                    b = est_bits(qt[by:by + cb, bx:bx + cb])
                    total += b
                    heat[y + by:y + by + cb, x + bx:x + bx + cb] = b / cb ** 2
    est_bytes = total / 8
    print(f"[6] entropy-coder ESTIMATE: {est_bytes:.0f} bytes "
          f"({img.size / max(est_bytes, 1):.2f}:1, {total / img.size:.3f} bits/pixel)")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
    im = ax[1].imshow(heat, cmap="magma"); ax[1].axis("off")
    ax[1].set_title("Estimated bits/coefficient per code-block (mosaic layout)")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    savefig(fig, a.out, "06_codeblock_cost.png")

    # ---- 7. Reconstruction ----
    def reconstruct(res=None, drop=0):
        out = np.zeros((H, W))
        for (y, x), qt in q.items():
            qt = qt.copy()
            if drop:                              # discard `drop` least-significant bit-planes
                qt = np.sign(qt) * ((np.abs(qt) >> drop) << drop)
            if res is not None:                   # keep resolutions 0..res only
                n = T >> (L - res)
                keep = np.zeros(qt.shape, bool); keep[:n, :n] = True
                qt = np.where(keep, qt, 0)
            out[y:y + T, x:x + T] = idwt2(qt, L)
        return np.clip(out + 128, 0, 255).astype(np.uint8)

    rec = reconstruct()
    identical = np.array_equal(img, rec)
    print(f"[7] reconstruction bit-exact: {identical}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original")
    ax[1].imshow(rec, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title(f"Decoded from integer coefficients\nbit-exact: {identical}")
    for x in ax: x.axis("off")
    savefig(fig, a.out, "07_reconstruction.png")

    # ---- 8. Resolution scalability + bit-plane truncation ----
    fig, axs = plt.subplots(1, L + 1, figsize=(4 * (L + 1), 4.2))
    for r, ax_ in enumerate(axs):
        ax_.imshow(reconstruct(res=r), cmap="gray", vmin=0, vmax=255)
        ax_.set_title(f"resolutions 0..{r}" + (" (LL only)" if r == 0 else "")); ax_.axis("off")
    savefig(fig, a.out, "08a_resolution_scalability.png")
    drops = [0, 1, 2, 3, 4]
    fig, axs = plt.subplots(1, len(drops), figsize=(3.6 * len(drops), 4.2))
    for d, ax_ in zip(drops, axs):
        r_ = reconstruct(drop=d)
        ax_.imshow(r_, cmap="gray", vmin=0, vmax=255)
        ax_.set_title(f"drop {d} LSB planes\nPSNR {psnr(img, r_):.1f} dB"); ax_.axis("off")
    savefig(fig, a.out, "08b_bitplane_truncation.png")

    try_openjpeg(img, a, est_bytes)
    print(f"\nFigures saved in ./{a.out}/")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
