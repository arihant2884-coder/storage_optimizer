import json
import math
import os
import xml.etree.ElementTree as ET
from typing import Any

import rasterio

from storage_optimizer.models import RasterMetadata


class AuxWriter:
    @staticmethod
    def _sanitize_nodata(value: Any) -> Any:
        """Converts non-standard JSON floats (-inf, inf, nan) to None or clean values."""
        if value is None:
            return None
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            return None
        return value

    @staticmethod
    def generate_aux_xml(filepath: str) -> str:
        """Generates a GDAL-compliant .aux.xml statistics sidecar file."""
        aux_path = f"{filepath}.aux.xml"

        with rasterio.open(filepath) as src:
            root = ET.Element("PAMDataset")

            for band_idx in range(1, src.count + 1):
                try:
                    b_stats = (
                        src.stats(band_idx, approx=False)
                        if hasattr(src, "stats")
                        else src.statistics(band_idx, approx=False)
                    )
                    s_min, s_max = b_stats.min, b_stats.max
                    s_mean, s_std = b_stats.mean, b_stats.std

                    pam_band = ET.SubElement(root, "PAMRasterBand", band=str(band_idx))
                    meta = ET.SubElement(pam_band, "Metadata")

                    ET.SubElement(meta, "MDI", key="STATISTICS_MINIMUM").text = str(
                        s_min
                    )
                    ET.SubElement(meta, "MDI", key="STATISTICS_MAXIMUM").text = str(
                        s_max
                    )
                    ET.SubElement(meta, "MDI", key="STATISTICS_MEAN").text = str(s_mean)
                    ET.SubElement(meta, "MDI", key="STATISTICS_STDDEV").text = str(
                        s_std
                    )
                except Exception:  # noqa: BLE001, S112
                    continue

            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)
            tree.write(aux_path, encoding="utf-8", xml_declaration=True)

        return aux_path

    @staticmethod
    def generate_stac_json(filepath: str, meta: RasterMetadata) -> str:
        """Generates a valid STAC 1.1.0 JSON metadata sidecar file."""
        stac_path = os.path.splitext(filepath)[0] + ".stac.json"
        filename = os.path.basename(filepath)

        # Extract numeric EPSG code (e.g., "EPSG:4326" -> 4326)
        epsg_code = None
        if meta.crs:
            crs_str = str(meta.crs)
            if "4326" in crs_str:
                epsg_code = 4326
            elif "EPSG:" in crs_str:
                try:
                    epsg_code = int(crs_str.split(":")[-1])
                except ValueError:
                    epsg_code = None

        sanitized_nodata = AuxWriter._sanitize_nodata(meta.nodata)

        stac_item: dict[str, Any] = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
                "https://stac-extensions.github.io/file/v2.0.0/schema.json",
            ],
            "id": os.path.splitext(filename)[0],
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [meta.bounds[0], meta.bounds[1]],
                        [meta.bounds[2], meta.bounds[1]],
                        [meta.bounds[2], meta.bounds[3]],
                        [meta.bounds[0], meta.bounds[3]],
                        [meta.bounds[0], meta.bounds[1]],
                    ]
                ],
            },
            "bbox": meta.bounds,
            "properties": {
                "datetime": "2026-01-01T00:00:00Z",
                "proj:epsg": epsg_code,
                "raster:bands": [
                    {
                        "data_type": meta.dtype,
                        "nodata": sanitized_nodata,
                    }
                    for _ in range(meta.count)
                ],
            },
            "assets": {
                "data": {
                    "href": f"./{filename}",
                    "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                    "roles": ["data"],
                    "file:size": os.path.getsize(filepath)
                    if os.path.exists(filepath)
                    else 0,
                }
            },
            "links": [],
        }

        with open(stac_path, "w", encoding="utf-8") as f:
            json.dump(stac_item, f, indent=2)

        return stac_path
