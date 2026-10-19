import os

import numpy as np
import rasterio

from storage_optimizer.models import CodecType, CompressionProfile, RasterMetadata


class CodecManager:
    @staticmethod
    def inspect_raster(src: rasterio.DatasetReader, filepath: str) -> RasterMetadata:
        dtype_str = str(src.dtypes[0])
        dtype_obj = np.dtype(dtype_str)
        is_complex = np.issubdtype(dtype_obj, np.complexfloating)
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        uncompressed_est = src.width * src.height * src.count * dtype_obj.itemsize
        is_large = file_size > (1.5 * 1024 * 1024 * 1024) or uncompressed_est > (
            2.0 * 1024 * 1024 * 1024
        )

        should_downcast = False
        if dtype_str == "float64":
            try:
                b_stats = (
                    src.stats(1, approx=True)
                    if hasattr(src, "stats")
                    else src.statistics(1, approx=True)
                )
                f32_min = float(np.finfo(np.float32).min)
                f32_max = float(np.finfo(np.float32).max)
                if b_stats.min >= f32_min and b_stats.max <= f32_max:
                    should_downcast = True
            except (AttributeError, TypeError, ValueError):
                should_downcast = False

        return RasterMetadata(
            filepath=filepath,
            driver=src.driver,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=dtype_str,
            crs=str(src.crs) if src.crs else None,
            bounds=[
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            nodata=src.nodata,
            file_size_bytes=file_size,
            is_complex=is_complex,
            is_large=is_large,
            should_downcast_float32=should_downcast,
        )

    @staticmethod
    def determine_profile(
        meta: RasterMetadata, max_z_error: float = 0.0
    ) -> CompressionProfile:
        dtype_obj = np.dtype(meta.dtype)
        is_float = np.issubdtype(dtype_obj, np.floating) or meta.is_complex
        is_int = np.issubdtype(dtype_obj, np.integer)

        target_dtype = "float32" if meta.should_downcast_float32 else None
        bigtiff_setting = "IF_SAFER" if meta.is_large else "IF_NEEDED"

        if meta.count in (3, 4) and meta.dtype in ("uint8", "byte"):
            return CompressionProfile(
                codec=CodecType.WEBP,
                predictor=1,
                zstd_level=12,
                block_size=512,
                num_threads="ALL_CPUS",
                bigtiff=bigtiff_setting,
                target_dtype=target_dtype,
                max_z_error=0.0,
            )

        if is_float and max_z_error > 0.0:
            return CompressionProfile(
                codec=CodecType.LERC_ZSTD,
                predictor=1,
                zstd_level=12,
                max_z_error=max_z_error,
                block_size=512,
                num_threads="ALL_CPUS",
                bigtiff=bigtiff_setting,
                target_dtype=target_dtype,
            )

        if is_int:
            return CompressionProfile(
                codec=CodecType.ZSTD,
                predictor=2,
                zstd_level=12,
                block_size=512,
                num_threads="ALL_CPUS",
                bigtiff=bigtiff_setting,
                target_dtype=target_dtype,
                max_z_error=0.0,
            )

        # High-Performance Purely Lossless Float (ZSTD + Predictor 3)
        return CompressionProfile(
            codec=CodecType.ZSTD,
            predictor=3 if is_float else 1,
            zstd_level=12,
            block_size=512,
            num_threads="ALL_CPUS",
            bigtiff=bigtiff_setting,
            target_dtype=target_dtype,
            max_z_error=0.0,
        )

    @staticmethod
    def auto_tune_candidates(
        meta: RasterMetadata, max_z_error: float = 0.0
    ) -> list[CompressionProfile]:
        dtype_obj = np.dtype(meta.dtype)
        is_float = np.issubdtype(dtype_obj, np.floating) or meta.is_complex
        is_int = np.issubdtype(dtype_obj, np.integer)
        target_dtype = "float32" if meta.should_downcast_float32 else None
        bigtiff_setting = "IF_SAFER" if meta.is_large else "IF_NEEDED"

        candidates = []

        if max_z_error > 0.0:
            candidates.append(
                CompressionProfile(
                    codec=CodecType.LERC_ZSTD,
                    predictor=1,
                    zstd_level=12,
                    max_z_error=max_z_error,
                    block_size=512,
                    num_threads="ALL_CPUS",
                    bigtiff=bigtiff_setting,
                    target_dtype=target_dtype,
                )
            )

        # Candidate: ZSTD + Predictor 3 (Max Lossless Float Optimization)
        if is_float:
            candidates.append(
                CompressionProfile(
                    codec=CodecType.ZSTD,
                    predictor=3,
                    zstd_level=12,
                    block_size=512,
                    num_threads="ALL_CPUS",
                    bigtiff=bigtiff_setting,
                    target_dtype=target_dtype,
                    max_z_error=0.0,
                )
            )

        candidates.append(
            CompressionProfile(
                codec=CodecType.ZSTD,
                predictor=2 if is_int else 1,
                zstd_level=12,
                block_size=512,
                num_threads="ALL_CPUS",
                bigtiff=bigtiff_setting,
                target_dtype=target_dtype,
                max_z_error=0.0,
            )
        )

        return candidates
