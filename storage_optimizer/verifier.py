from __future__ import annotations

import numpy as np
import rasterio

from storage_optimizer.models import VerificationResult


class RasterVerifier:
    @staticmethod
    def verify_parity(
        source_path: str,
        compressed_path: str,
        max_allowed_error: float = 0.0,
        target_dtype: str | None = None,
    ) -> VerificationResult:
        """Verifies pixel parity via streaming block windows with NaN tolerance."""
        mismatched_windows = 0
        total_windows = 0
        max_observed_error = 0.0

        with rasterio.open(source_path) as src, rasterio.open(compressed_path) as dst:
            src_size = (
                src.width * src.height * src.count * np.dtype(src.dtypes[0]).itemsize
            )
            dst_size = rasterio.os.path.getsize(compressed_path)

            first_band_block = src.block_shapes[0]
            block_w, block_h = first_band_block[1], first_band_block[0]

            for j in range(0, src.height, block_h):
                for i in range(0, src.width, block_w):
                    total_windows += 1
                    w_width = min(block_w, src.width - i)
                    w_height = min(block_h, src.height - j)
                    win = rasterio.windows.Window(i, j, w_width, w_height)

                    src_data = src.read(window=win)
                    dst_data = dst.read(window=win)

                    if target_dtype and str(src_data.dtype) != target_dtype:
                        src_data = src_data.astype(target_dtype)

                    # Build comparison mask excluding NaNs and explicit NoData
                    mask = np.ones_like(src_data, dtype=bool)

                    if np.issubdtype(src_data.dtype, np.floating):
                        mask &= ~np.isnan(src_data) & ~np.isnan(dst_data)

                    if src.nodata is not None and not np.isnan(src.nodata):
                        mask &= (src_data != src.nodata) & (dst_data != src.nodata)

                    if not np.any(mask):
                        continue

                    valid_src = src_data[mask]
                    valid_dst = dst_data[mask]

                    diff = np.abs(
                        valid_src.astype(np.float64) - valid_dst.astype(np.float64)
                    )
                    window_max_err = float(np.max(diff)) if diff.size > 0 else 0.0

                    max_observed_error = max(max_observed_error, window_max_err)

                    effective_max_allowed = (
                        max_allowed_error + 1e-6 if max_allowed_error > 0 else 0.0
                    )

                    if effective_max_allowed == 0.0:
                        if not np.allclose(
                            valid_src, valid_dst, rtol=1e-5, atol=1e-7, equal_nan=True
                        ):
                            mismatched_windows += 1
                    else:
                        if window_max_err > effective_max_allowed:
                            mismatched_windows += 1

        is_valid = mismatched_windows == 0
        return VerificationResult(
            is_valid=is_valid,
            mismatched_windows=mismatched_windows,
            total_windows=total_windows,
            max_observed_error=max_observed_error,
            max_allowed_error=max_allowed_error,
            source_size_bytes=src_size,
            compressed_size_bytes=dst_size,
        )
