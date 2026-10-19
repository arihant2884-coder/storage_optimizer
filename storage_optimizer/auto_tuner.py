import tempfile

import rasterio
from rasterio.windows import Window

from storage_optimizer.codecs import CodecManager
from storage_optimizer.models import CompressionProfile, RasterMetadata


class AutoTuner:
    @staticmethod
    def benchmark_sample_window(
        src_path: str, meta: RasterMetadata, max_z_error: float = 0.0
    ) -> CompressionProfile:
        """Benchmarks candidate profiles on a sample raster window to select the highest compression ratio."""
        candidates = CodecManager.auto_tune_candidates(meta, max_z_error=max_z_error)
        if len(candidates) == 1:
            return candidates[0]

        best_profile = candidates[0]
        best_size = float("inf")

        with rasterio.open(src_path) as src:
            sample_w = min(2048, src.width)
            sample_h = min(2048, src.height)
            sample_win = Window(
                src.width // 2 - sample_w // 2,
                src.height // 2 - sample_h // 2,
                sample_w,
                sample_h,
            )
            data = src.read(window=sample_win)

            for profile in candidates:
                out_dtype = profile.target_dtype or src.dtypes[0]
                opts = profile.to_gdal_options()

                with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                    tmp_profile = {
                        "driver": "GTiff",
                        "height": sample_h,
                        "width": sample_w,
                        "count": src.count,
                        "dtype": out_dtype,
                        "crs": src.crs,
                        "transform": rasterio.windows.transform(
                            sample_win, src.transform
                        ),
                        **opts,
                    }

                    try:
                        with rasterio.open(tmp.name, "w", **tmp_profile) as dst:
                            if (
                                profile.target_dtype
                                and str(data.dtype) != profile.target_dtype
                            ):
                                dst.write(data.astype(profile.target_dtype))
                            else:
                                dst.write(data)
                        compressed_size = tempfile.os.path.getsize(tmp.name)

                        if compressed_size < best_size:
                            best_size = compressed_size
                            best_profile = profile
                    except Exception:  # noqa: BLE001, S112
                        continue

        return best_profile
