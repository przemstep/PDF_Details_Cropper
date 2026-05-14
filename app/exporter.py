from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import safe_filename


class Exporter:
    def export(self, extraction_xlsx: Path, classification_xlsx: Path, output_dir: Path, project_name: str) -> Path:
        ex = pd.read_excel(extraction_xlsx)
        cl = pd.read_excel(classification_xlsx)
        merged = ex.merge(cl, on="crop_id", how="left") if "crop_id" in ex.columns else ex

        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for row in merged.itertuples(index=False):
            code = getattr(row, "proposed_code", "UNCLASSIFIED")
            page = getattr(row, "page", 0)
            crop_id = getattr(row, "crop_id", "NA")
            filename = safe_filename(f"DET_{code}_REF-{project_name}_S{page}_{crop_id}.pdf")
            records.append({**row._asdict(), "FileName": filename, "Status": "Do weryfikacji"})

        meta = pd.DataFrame(records)
        metadata_path = output_dir / "metadata.xlsx"
        meta.to_excel(metadata_path, index=False)
        return metadata_path
