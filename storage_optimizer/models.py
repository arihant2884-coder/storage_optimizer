from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CodecType(str, Enum):
    ZSTD = "zstd"
    WEBP = "webp"
    LZW = "lzw"
    DEFLATE = "deflate"
    LERC_ZSTD = "lerc_zstd"
    NONE = "none"


class OptimizationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RasterMetadata:
    filepath: str
    driver: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None
    bounds: list[float]
    nodata: float | None
    file_size_bytes: int
    is_complex: bool
    is_large: bool
    should_downcast_float32: bool = False


@dataclass
class CompressionProfile:
    codec: CodecType
    predictor: int = 1
    zstd_level: int = 9
    block_size: int = 1024
    num_threads: str = "ALL_CPUS"
    bigtiff: str = "IF_SAFER"
    target_dtype: str | None = None
    max_z_error: float = 0.0

    def to_gdal_options(self) -> dict[str, Any]:
        """Converts profile parameters into driver-specific GDAL creation options."""
        opts: dict[str, Any] = {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": self.block_size,
            "blockysize": self.block_size,
            "compress": self.codec.value.upper(),
            "num_threads": self.num_threads,
            "bigtiff": self.bigtiff,
            "sparse_ok": True,
        }

        if self.codec == CodecType.ZSTD:
            opts["zstd_level"] = self.zstd_level
            opts["predictor"] = self.predictor
        elif self.codec == CodecType.LERC_ZSTD:
            opts["zstd_level"] = self.zstd_level
            opts["max_z_error"] = self.max_z_error
        elif self.codec == CodecType.WEBP:
            opts["webp_lossless"] = "YES"
        elif self.codec in (CodecType.LZW, CodecType.DEFLATE):
            opts["predictor"] = self.predictor
            if self.codec == CodecType.DEFLATE:
                opts["zlevel"] = 9

        return opts


@dataclass
class VerificationResult:
    is_valid: bool
    mismatched_windows: int
    total_windows: int
    max_observed_error: float
    max_allowed_error: float
    source_size_bytes: int
    compressed_size_bytes: int

    @property
    def space_savings_percent(self) -> float:
        if self.source_size_bytes == 0:
            return 0.0
        return (1.0 - (self.compressed_size_bytes / self.source_size_bytes)) * 100.0


@dataclass
class OptimizationResult:
    success: bool
    status: OptimizationStatus
    source_path: str
    output_path: str
    metadata: RasterMetadata | None = None
    profile_used: CompressionProfile | None = None
    verification: VerificationResult | None = None
    elapsed_seconds: float = 0.0
    error_message: str | None = None
