"""Objective visual-complexity metrics for figure images/files."""
import numpy as np


def image_metrics(png_path):
    """-> dict of raster visual-complexity metrics."""
    from PIL import Image
    import os
    from scipy import ndimage
    g = np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
    ink = g < 0.6
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    hist, _ = np.histogram(g, bins=16, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    lab, ncomp = ndimage.label(ink)
    return dict(
        ink_ratio=float(ink.mean()),
        edge_density=float((gx > 0.2).mean() + (gy > 0.2).mean()) / 2,
        grad_energy=float((gx ** 2).mean() + (gy ** 2).mean()),
        components=int(ncomp),
        entropy=float(-(p * np.log2(p)).sum()),
        contrast=float(g.std()),
        png_bytes=os.path.getsize(png_path),
    )


def svg_metrics(svg_path):
    """-> dict of vector-complexity metrics."""
    import os, re
    s = open(svg_path).read()
    return dict(
        svg_bytes=os.path.getsize(svg_path),
        svg_paths=len(re.findall(r"<path", s)),
        svg_uses=len(re.findall(r"<use", s)),
        svg_elements=len(re.findall(r"<(path|use|g|text|defs)", s)),
    )


def ssim(a_path, b_path):
    """Grayscale SSIM (single-window global) between two same-size PNGs."""
    from PIL import Image
    a = np.asarray(Image.open(a_path).convert("L"), dtype=np.float64) / 255.0
    b = np.asarray(Image.open(b_path).convert("L"), dtype=np.float64) / 255.0
    if a.shape != b.shape:
        b_img = Image.fromarray((b * 255).astype(np.uint8))
        b_img = b_img.resize((a.shape[1], a.shape[0]))
        b = np.asarray(b_img, dtype=np.float64) / 255.0
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + C1) * (2 * cov + C2))
                 / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))
