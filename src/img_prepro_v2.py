"""Robust preprocessing for GONG H-alpha images (EdgeAttNet-style pipeline, hardened).

Changes vs. naive version:
- Disk detection: Otsu centroid -> polar unwrap -> limb = steepest intensity drop per ray,
  searched only within a narrow band around the known solar radius -> robust least-squares
  circle fit. The band is what makes it find the true limb rather than the outer edge of an
  atmospheric halo, which is often the stronger edge of the two.
- Mask shrunk to `limb_margin` * R to drop the unreliable limb rim.
- Background: 1D radial profile, then 2D large-scale median correction (clouds, gradients).
- Fixed global ratio window for 8-bit conversion -> consistent brightness across images.
- Mild CLAHE applied with a neutral off-disk fill.
"""
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic


class DiskDetectionError(RuntimeError):
    pass


class ImgPreProcess:
    def __init__(
        self,
        image_path: str | Path,
        limb_margin: float = 0.98,     # keep pixels with r <= limb_margin * R
        reference_radius: float = 907.0,
        radius_tolerance: float = 0.05,
        n_bins: int = 100,
        bg_scale: int = 8,             # downsample factor for 2D background
        bg_kernel: int = 31,           # median kernel at downsampled scale (~250 px full-res)
        ratio_window: tuple[float, float] = (0.6, 1.4),  # fixed, same for every image
        use_clahe: bool = True,
        clahe_clip: float = 1.5,
        clahe_grid: int = 8,
    ) -> None:
        self.image_path = Path(image_path)
        image = cv.imread(str(self.image_path), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {self.image_path}")
        self.image_u8 = image
        self.limb_margin = limb_margin
        self.reference_radius = reference_radius
        self.radius_tolerance = radius_tolerance
        self.n_bins = n_bins
        self.bg_scale = bg_scale
        self.bg_kernel = bg_kernel
        self.ratio_window = ratio_window
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.stages: dict[str, np.ndarray] = {}   # for debugging: inspect each step
        self.disk: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------ pipeline
    def run(self) -> np.ndarray:
        img = self.image_u8.astype(np.float32) / 255.0
        cx, cy, r = self.detect_disk(self.image_u8)
        self.disk = (cx, cy, r)

        yy, xx = np.indices(img.shape)
        rho = np.hypot(xx - cx, yy - cy) / r
        mask = rho <= self.limb_margin

        radial_flat = self.radial_flattening(img, mask, rho)
        flat = self.large_scale_correction(radial_flat, mask)
        u8 = self.to_uint8_fixed(flat, mask)
        smooth = cv.GaussianBlur(u8, (3, 3), sigmaX=0.7, sigmaY=0.7)
        out = self.clahe_in_disk(smooth, mask) if self.use_clahe else smooth
        out[~mask] = 0

        self.stages = {"radial_flat": radial_flat, "flat": flat, "u8": u8, "out": out}
        return out

    # ------------------------------------------------------------ disk detection
    def detect_disk(self, image_u8: np.ndarray) -> tuple[float, float, float]:
        """Locate the solar limb.

        The search is restricted to a narrow band around `reference_radius`.
        Every frame in this dataset comes from GONG at 2048x2048, and the Sun's
        angular radius varies by only about +/-1.7% over a year, so the limb can
        only ever fall in a narrow band. Without that constraint, images with an
        atmospheric halo around the disk defeat a steepest-gradient search: the
        halo's outer edge is a stronger edge than the limb, and the detector
        locked onto it on 14% of the training set, reporting r=1036 instead of
        r=907 and leaving a ring artefact in the output.
        """
        h, w = image_u8.shape

        # 1) Rough centre: Otsu + largest connected component. Only the centroid
        #    is used - the blob's area is inflated by any halo, so its radius is
        #    not trustworthy.
        blur = cv.GaussianBlur(image_u8, (0, 0), 5)
        _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        n, labels, stats, cents = cv.connectedComponentsWithStats(binary)
        if n < 2:
            raise DiskDetectionError(f"No bright region found in {self.image_path}")
        k = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        cx0, cy0 = cents[k]

        # 2) Polar unwrap around the rough centre; limb = steepest drop per ray,
        #    searched only within the plausible band.
        n_theta = 720
        max_r = int(min(cx0, cy0, w - cx0, h - cy0)) - 2
        lo = max(1, int(self.reference_radius * (1.0 - self.radius_tolerance)))
        hi = min(max_r - 1, int(self.reference_radius * (1.0 + self.radius_tolerance)))
        if hi <= lo:
            raise DiskDetectionError(
                f"Radius band [{lo}, {hi}] does not fit the frame in {self.image_path}"
            )

        polar = cv.warpPolar(
            blur.astype(np.float32), (max_r, n_theta), (cx0, cy0), max_r,
            cv.WARP_POLAR_LINEAR + cv.INTER_LINEAR,
        )  # rows = angle, cols = radius in pixels
        grad = np.gradient(polar, axis=1)
        limb_r = lo + np.argmin(grad[:, lo:hi], axis=1)      # most negative derivative
        depth = -grad[np.arange(n_theta), limb_r]
        theta = np.arange(n_theta) * 2 * np.pi / n_theta
        px = cx0 + limb_r * np.cos(theta)
        py = cy0 + limb_r * np.sin(theta)

        # Drop rays whose edge is weak - cloud, or a limb that never darkens
        keep = depth > 0.25 * np.median(depth)
        if keep.sum() < 50:
            raise DiskDetectionError(f"No limb edge found in {self.image_path}")

        # 3) Robust circle fit with iterative outlier rejection
        cx, cy, r = self._fit_circle(px[keep], py[keep])
        for _ in range(3):
            resid = np.abs(np.hypot(px[keep] - cx, py[keep] - cy) - r)
            mad = np.median(resid) + 1e-6
            good = resid < 3.0 * 1.4826 * mad + 1.0
            idx = np.flatnonzero(keep)
            keep[idx[~good]] = False
            if keep.sum() < 50:
                raise DiskDetectionError(f"Too few limb points in {self.image_path}")
            cx, cy, r = self._fit_circle(px[keep], py[keep])

        drift = abs(r - self.reference_radius) / self.reference_radius
        if drift > self.radius_tolerance:
            raise DiskDetectionError(
                f"Radius {r:.1f} is {100 * drift:.1f}% from the reference "
                f"{self.reference_radius:.0f} in {self.image_path}"
            )
        return float(cx), float(cy), float(r)

    @staticmethod
    def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        # Kasa algebraic fit: x^2 + y^2 + D x + E y + F = 0
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x**2 + y**2)
        (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -D / 2, -E / 2
        return cx, cy, float(np.sqrt(cx**2 + cy**2 - F))

    # --------------------------------------------------------------- background
    def radial_flattening(self, img: np.ndarray, mask: np.ndarray, rho: np.ndarray) -> np.ndarray:
        stats = binned_statistic(rho[mask], img[mask], statistic="median",
                                 bins=self.n_bins, range=(0.0, self.limb_margin))
        centers = 0.5 * (stats.bin_edges[:-1] + stats.bin_edges[1:])
        valid = ~np.isnan(stats.statistic)
        if valid.sum() < 7:
            raise DiskDetectionError(f"Radial profile empty for {self.image_path}")
        profile = savgol_filter(stats.statistic[valid], window_length=7, polyorder=2)
        profile = np.clip(profile, 1e-3, None)
        background = np.interp(rho, centers[valid], profile).astype(np.float32)

        flat = np.ones_like(img, dtype=np.float32)
        flat[mask] = img[mask] / background[mask]
        return flat

    def large_scale_correction(self, flat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove non-radial large-scale variations (clouds, gradients) with a coarse median."""
        h, w = flat.shape
        s = self.bg_scale
        lo, hi = 0.5, 1.5
        filled = np.where(mask, flat, 1.0)                       # neutral outside disk
        small = cv.resize(filled, (w // s, h // s), interpolation=cv.INTER_AREA)
        small_u8 = np.clip((small - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        bg_small = cv.medianBlur(small_u8, self.bg_kernel).astype(np.float32) / 255 * (hi - lo) + lo
        bg = cv.resize(bg_small, (w, h), interpolation=cv.INTER_CUBIC)
        bg = np.clip(bg, 0.3, None)

        out = np.ones_like(flat)
        out[mask] = flat[mask] / bg[mask]
        return out

    # ---------------------------------------------------------- intensity mapping
    def to_uint8_fixed(self, flat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lo, hi = self.ratio_window
        out = (np.clip((flat - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        out[~mask] = 0
        return out

    def clahe_in_disk(self, img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = img_u8.copy()
        filled[~mask] = 128          # ratio 1.0 maps to ~128: neutral, same for every image
        clahe = cv.createCLAHE(clipLimit=self.clahe_clip,
                               tileGridSize=(self.clahe_grid, self.clahe_grid))
        return clahe.apply(filled)


def debug_dump(image_path: str | Path, out_dir: str | Path) -> None:
    """Save every stage + detected limb overlay for a single image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = ImgPreProcess(image_path)
    p.run()
    stem = Path(image_path).stem
    cx, cy, r = p.disk
    overlay = cv.cvtColor(p.image_u8, cv.COLOR_GRAY2BGR)
    cv.circle(overlay, (round(cx), round(cy)), round(r), (0, 0, 255), 2)
    cv.imwrite(str(out_dir / f"{stem}_0_limb.png"), overlay)
    for i, key in enumerate(["radial_flat", "flat"], start=1):
        vis = np.clip((p.stages[key] - 0.6) / 0.8 * 255, 0, 255).astype(np.uint8)
        cv.imwrite(str(out_dir / f"{stem}_{i}_{key}.png"), vis)
    cv.imwrite(str(out_dir / f"{stem}_3_out.png"), p.stages["out"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a directory of GONG H-alpha frames.")
    parser.add_argument("src", type=Path, help="directory of .jpg/.jpeg frames")
    parser.add_argument("dst", type=Path, help="where to write the processed frames")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    failures = []
    radii = []
    for f in sorted(args.src.glob("*.jp*g")):
        try:
            p = ImgPreProcess(f)
            cv.imwrite(str(args.dst / f.name), p.run())
            radii.append(p.disk[2])
        except DiskDetectionError as e:
            failures.append((f.name, str(e)))

    print(f"processed {len(radii)}, {len(failures)} failures")
    if radii:
        radii = np.array(radii)
        print(f"radius: median={np.median(radii):.1f} min={radii.min():.1f} max={radii.max():.1f}")
    for name, msg in failures:
        print(" ", name, msg)