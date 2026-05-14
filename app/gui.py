from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from .classifier import DetailClassifier
from .exporter import Exporter
from .pdf_extractor import PDFExtractor
from .synonym_builder import SynonymBuilder


class AppGUI(tk.Tk):
    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.title("PDF Details Cropper")
        self.geometry("1080x720")
        self.root_dir = root_dir
        self.source_pdf: Path | None = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.extract_tab = ttk.Frame(nb)
        self.dict_tab = ttk.Frame(nb)
        self.class_tab = ttk.Frame(nb)
        self.export_tab = ttk.Frame(nb)
        nb.add(self.extract_tab, text="Extract")
        nb.add(self.dict_tab, text="Dictionary Builder")
        nb.add(self.class_tab, text="Classification")
        nb.add(self.export_tab, text="Export")

        self._build_extract_tab()
        self._build_dict_tab()
        self._build_class_tab()
        self._build_export_tab()

    def _build_extract_tab(self) -> None:
        ttk.Button(self.extract_tab, text="Wybierz PDF", command=self._select_pdf).pack(pady=8)
        ttk.Button(self.extract_tab, text="Start Extract", command=self._run_extract).pack(pady=8)

    def _build_dict_tab(self) -> None:
        ttk.Button(self.dict_tab, text="Generate dictionary template", command=self._gen_dict).pack(pady=8)
        ttk.Button(self.dict_tab, text="Generate frequency analysis", command=self._freq).pack(pady=8)

    def _build_class_tab(self) -> None:
        ttk.Button(self.class_tab, text="Run classification", command=self._classify).pack(pady=8)

    def _build_export_tab(self) -> None:
        ttk.Button(self.export_tab, text="Export metadata", command=self._export).pack(pady=8)

    def _select_pdf(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.source_pdf = Path(p)

    def _run_extract(self) -> None:
        if not self.source_pdf:
            return
        PDFExtractor(self.root_dir / "intermediate").extract(self.source_pdf, self.source_pdf.stem)
        messagebox.showinfo("Done", "Extraction completed")

    def _gen_dict(self) -> None:
        SynonymBuilder(self.root_dir / "data/dictionaries/master_dictionary.xlsx").ensure_dictionary_template()
        messagebox.showinfo("Done", "Dictionary template generated")

    def _freq(self) -> None:
        sb = SynonymBuilder(self.root_dir / "data/dictionaries/master_dictionary.xlsx")
        freq = sb.analyze_extractions(self.root_dir / "intermediate/extraction_data.xlsx")
        freq.to_excel(self.root_dir / "data/dictionaries/frequency_candidates.xlsx", index=False)
        messagebox.showinfo("Done", "Frequency analysis exported")

    def _classify(self) -> None:
        DetailClassifier().classify(
            self.root_dir / "intermediate/extraction_data.xlsx",
            self.root_dir / "data/dictionaries/master_dictionary.xlsx",
            self.root_dir / "intermediate",
        )
        messagebox.showinfo("Done", "Classification completed")

    def _export(self) -> None:
        Exporter().export(
            self.root_dir / "intermediate/extraction_data.xlsx",
            self.root_dir / "intermediate/classification_results.xlsx",
            self.root_dir / "output",
            "PROJECT",
        )
        messagebox.showinfo("Done", "Metadata exported")
