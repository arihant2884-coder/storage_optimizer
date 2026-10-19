from __future__ import annotations

import os
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

from storage_optimizer.auto_tuner import AutoTuner
from storage_optimizer.aux_writer import AuxWriter
from storage_optimizer.codecs import CodecManager
from storage_optimizer.models import (
    CompressionProfile,
    OptimizationResult,
    OptimizationStatus,
)
from storage_optimizer.verifier import RasterVerifier


class RasterOptimizer:
    def __init__(self, generate_aux: bool = True, generate_stac: bool = True):
        self.generate_aux = generate_aux
        self.generate_stac = generate_stac

    def optimize(
        self,
        input_path: str,
        output_path: str,
        auto_tune: bool = True,
        max_z_error: float = 0.0,
        custom_profile: CompressionProfile | None = None,
    ) -> OptimizationResult:
        start_time = time.time()

        if not os.path.exists(input_path):
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.FAILED,
                source_path=input_path,
                output_path=output_path,
                error_message=f"Input path '{input_path}' does not exist.",
            )

        with rasterio.open(input_path) as src:
            meta = CodecManager.inspect_raster(src, input_path)

            if custom_profile:
                profile = custom_profile
            elif auto_tune:
                profile = AutoTuner.benchmark_sample_window(
                    input_path, meta, max_z_error=max_z_error
                )
            else:
                profile = CodecManager.determine_profile(meta, max_z_error=max_z_error)

            out_dtype = profile.target_dtype or src.dtypes[0]
            out_profile = src.profile.copy()
            gdal_opts = profile.to_gdal_options()

            out_profile.update(
                dtype=out_dtype,
                count=src.count,
                width=src.width,
                height=src.height,
                **gdal_opts,
            )

            block_w = profile.block_size
            block_h = profile.block_size
            windows = [
                rasterio.windows.Window(
                    col,
                    row,
                    min(block_w, src.width - col),
                    min(block_h, src.height - row),
                )
                for row in range(0, src.height, block_h)
                for col in range(0, src.width, block_w)
            ]

            with rasterio.open(output_path, "w", **out_profile) as dst:
                for win in tqdm(
                    windows, desc="Compressing Raster Windows", unit="block"
                ):
                    data = src.read(window=win)
                    if profile.target_dtype and str(data.dtype) != profile.target_dtype:
                        data = data.astype(profile.target_dtype)
                    dst.write(data, window=win)

                if meta.is_large or src.width > 4096:
                    factors = [2, 4, 8, 16]
                    resampling = (
                        Resampling.average
                        if np.issubdtype(np.dtype(out_dtype), np.floating)
                        else Resampling.nearest
                    )
                    dst.build_overviews(factors, resampling)
                    dst.update_tags(ns="rio_overview", resampling=resampling.name)

        verifier = RasterVerifier()
        verification = verifier.verify_parity(
            source_path=input_path,
            compressed_path=output_path,
            max_allowed_error=profile.max_z_error,
            target_dtype=profile.target_dtype,
        )

        if self.generate_aux:
            AuxWriter.generate_aux_xml(output_path)

        if self.generate_stac:
            AuxWriter.generate_stac_json(output_path, meta)

        elapsed = time.time() - start_time

        return OptimizationResult(
            success=verification.is_valid,
            status=OptimizationStatus.SUCCESS
            if verification.is_valid
            else OptimizationStatus.FAILED,
            source_path=input_path,
            output_path=output_path,
            metadata=meta,
            profile_used=profile,
            verification=verification,
            elapsed_seconds=elapsed,
        )
