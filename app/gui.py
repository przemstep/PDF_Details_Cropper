from __future__ import annotations

import threading
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
        self.extract_output_dir: Path = self.root_dir / "intermediate"
        self.is_extracting = False

        self.pdf_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(self.extract_output_dir))
        self.status_var = tk.StringVar(value="Status: idle")

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
        frame = ttk.Frame(self.extract_tab, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Wybrany plik PDF").pack(anchor="w")
        pdf_row = ttk.Frame(frame)
        pdf_row.pack(fill="x", pady=(2, 10))
        ttk.Entry(pdf_row, textvariable=self.pdf_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(pdf_row, text="Wybierz", command=self._select_pdf).pack(side="left", padx=(8, 0))

        ttk.Label(frame, text="Katalog output").pack(anchor="w")
        out_row = ttk.Frame(frame)
        out_row.pack(fill="x", pady=(2, 10))
        ttk.Entry(out_row, textvariable=self.output_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Wybierz", command=self._select_output_dir).pack(side="left", padx=(8, 0))

        self.extract_button = ttk.Button(frame, text="Start Extract", command=self._run_extract)
        self.extract_button.pack(anchor="w", pady=(0, 10))

        ttk.Label(frame, textvariable=self.status_var).pack(anchor="w", pady=(0, 10))

        ttk.Label(frame, text="Błędy").pack(anchor="w")
        self.error_box = tk.Text(frame, height=8, state="disabled")
        self.error_box.pack(fill="both", expand=True)

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
            self.pdf_path_var.set(str(self.source_pdf))

    def _select_output_dir(self) -> None:
        p = filedialog.askdirectory(initialdir=str(self.extract_output_dir))
        if p:
            self.extract_output_dir = Path(p)
            self.output_dir_var.set(str(self.extract_output_dir))

    def _append_error(self, message: str) -> None:
        self.error_box.configure(state="normal")
        self.error_box.insert("end", f"{message}\n")
        self.error_box.see("end")
        self.error_box.configure(state="disabled")

    def _set_extract_running(self, running: bool) -> None:
        self.is_extracting = running
        self.extract_button.configure(state="disabled" if running else "normal")

    def _run_extract(self) -> None:
        if self.is_extracting:
            return

        pdf_path = self.pdf_path_var.get().strip()
        out_path = self.output_dir_var.get().strip()
        if not pdf_path:
            self._append_error("Nie wybrano pliku PDF.")
            return
        if not out_path:
            self._append_error("Nie wybrano katalogu output.")
            return

        self.source_pdf = Path(pdf_path)
        self.extract_output_dir = Path(out_path)

        self._set_extract_running(True)
        self.status_var.set("Status: analyzing...")

        def worker() -> None:
            try:
                records = PDFExtractor(self.extract_output_dir).extract(self.source_pdf, self.source_pdf.stem)
                page_count = len({r.page for r in records})
                box_count = len(records)
                self.after(0, lambda: self.status_var.set(f"Analyzed: {page_count} pages, {box_count} boxes"))
                self.after(0, lambda: messagebox.showinfo("Done", "Extraction completed"))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._append_error(str(exc)))
                self.after(0, lambda: self.status_var.set("Status: failed"))
            finally:
                self.after(0, lambda: self._set_extract_running(False))

        threading.Thread(target=worker, daemon=True).start()

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
