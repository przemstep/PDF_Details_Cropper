from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import pandas as pd

from .classifier import DetailClassifier
from .exporter import Exporter
from .ocr_runtime import OCRRuntime
from .pdf_extractor import PDFExtractor
from .settings import AppSettings, load_settings, save_settings
from .synonym_builder import SynonymBuilder
from .synonym_candidates import SynonymCandidateService
from .utils import ensure_log_dir


class AppGUI(tk.Tk):
    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.title("PDF Details Cropper")
        self.geometry("1080x720")
        self.root_dir = root_dir
        self.source_pdf: Path | None = None
        self.settings = load_settings(root_dir)
        self.extract_output_dir: Path = self.root_dir / "intermediate"
        self.is_extracting = False

        self.pdf_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(self.extract_output_dir))
        self.crop_padding_var = tk.StringVar(value="10")
        self.tesseract_path_var = tk.StringVar(value=self.settings.tesseract_exe_path)
        self.ocr_lang_var = tk.StringVar(value=self.settings.ocr_languages)
        self.ocr_status_var = tk.StringVar(value="OCR status: unknown")
        self.dictionary_path_var = tk.StringVar(value="")
        self.status_vars = {}
        self.page_status_vars = {}
        self.bbox_status_vars = {}
        self.count_status_vars = {}
        self.path_status_vars = {}
        self.stage_vars = {}
        self.tab_logs = {}
        self.logger = logging.getLogger(__name__)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.extract_tab = ttk.Frame(nb)
        self.dict_tab = ttk.Frame(nb)
        self.class_tab = ttk.Frame(nb)
        self.export_tab = ttk.Frame(nb)
        nb.add(self.extract_tab, text="Extract")
        nb.add(self.dict_tab, text="Analyze Text")
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

        ttk.Label(frame, text="Crop padding [pt]").pack(anchor="w")
        ttk.Entry(frame, textvariable=self.crop_padding_var).pack(anchor="w", pady=(2, 10))


        ocr_box = ttk.LabelFrame(frame, text="OCR / Tesseract", padding=8)
        ocr_box.pack(fill="x", pady=(0, 10))
        ttk.Label(ocr_box, text="Ścieżka tesseract.exe").pack(anchor="w")
        ocr_row = ttk.Frame(ocr_box)
        ocr_row.pack(fill="x", pady=(2, 6))
        ttk.Entry(ocr_row, textvariable=self.tesseract_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(ocr_row, text="Wskaż Tesseract", command=self._select_tesseract).pack(side="left", padx=(8, 0))
        ttk.Button(ocr_row, text="Test OCR", command=self._test_ocr).pack(side="left", padx=(8, 0))
        lang_row = ttk.Frame(ocr_box)
        lang_row.pack(fill="x", pady=(2, 6))
        ttk.Label(lang_row, text="Języki OCR:").pack(side="left")
        ttk.Entry(lang_row, textvariable=self.ocr_lang_var, width=20).pack(side="left", padx=(8, 0))
        ttk.Button(lang_row, text="Zapisz OCR", command=self._save_ocr_settings).pack(side="left", padx=(8, 0))
        ttk.Label(ocr_box, textvariable=self.ocr_status_var).pack(anchor="w")

        self.extract_button = ttk.Button(frame, text="Start Extract", command=self._run_extract)
        self.extract_button.pack(anchor="w", pady=(0, 10))

        self._build_status_panel(frame, "extract")

        self._create_tab_description(
            frame,
            "Extract",
            [
                "Co robi: Analizuje PDF i wycina bbox do plików PDF/PNG/TXT.",
                "Dane wejściowe: Plik PDF oraz folder roboczy.",
                "Dane wyjściowe: extraction_data.xlsx/json, wycięte szczegóły i podglądy.",
                "Miejsce zapisu: Folder roboczy ustawiony w tej zakładce (z podfolderami).",
                "Uwagi: To główny folder roboczy dla pozostałych funkcji, o ile funkcja nie pozwala wybrać innego.",
            ],
        )


    def _build_dict_tab(self) -> None:
        self._create_tab_description(
            self.dict_tab,
            "Analyze Text",
            [
                "Co robi: Buduje słownik i analizę częstotliwości z danych ekstrakcji.",
                "Dane wejściowe: extraction_data.xlsx z folderu roboczego.",
                "Dane wyjściowe: pliki słownika i kandydatów częstotliwości.",
                "Miejsce zapisu: folder roboczy (podfolder dictionaries).",
                "Uwagi: Korzysta z folderu ustawionego w Extract.",
            ],
        )
        dict_row = ttk.Frame(self.dict_tab)
        dict_row.pack(fill="x", padx=8, pady=8)
        ttk.Entry(dict_row, textvariable=self.dictionary_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(dict_row, text="Wybierz master_dictionary.xlsx", command=self._select_dictionary).pack(side="left", padx=(8, 0))
        ttk.Button(self.dict_tab, text="Analyze text", command=self._freq).pack(pady=8)
        ttk.Button(self.dict_tab, text="Generate synonym candidates", command=self._generate_synonym_candidates).pack(pady=4)
        ttk.Button(self.dict_tab, text="Import accepted synonyms", command=self._import_accepted_synonyms).pack(pady=4)
        self._build_status_panel(self.dict_tab, "dictionary")

    def _build_class_tab(self) -> None:
        self._create_tab_description(
            self.class_tab,
            "Classification",
            [
                "Co robi: Klasyfikuje rekordy ekstrakcji na podstawie słownika.",
                "Dane wejściowe: extraction_data.xlsx oraz master_dictionary.xlsx.",
                "Dane wyjściowe: classification_results.xlsx.",
                "Miejsce zapisu: folder roboczy ustawiony w Extract.",
                "Uwagi: Korzysta z tego samego folderu roboczego.",
            ],
        )
        ttk.Button(self.class_tab, text="Run classification", command=self._classify).pack(pady=8)
        self._build_status_panel(self.class_tab, "classification")

    def _build_export_tab(self) -> None:
        self._create_tab_description(
            self.export_tab,
            "Export",
            [
                "Co robi: Eksportuje metadane projektu na podstawie ekstrakcji i klasyfikacji.",
                "Dane wejściowe: extraction_data.xlsx i classification_results.xlsx.",
                "Dane wyjściowe: pliki eksportu metadanych.",
                "Miejsce zapisu: folder roboczy / exports.",
                "Uwagi: Domyślnie używa folderu z Extract.",
            ],
        )
        ttk.Button(self.export_tab, text="Export metadata", command=self._export).pack(pady=8)
        self._build_status_panel(self.export_tab, "export")

    def _create_tab_description(self, parent: tk.Widget, function_name: str, bullets: list[str]) -> None:
        box = ttk.LabelFrame(parent, text=f"Funkcja: {function_name}", padding=8)
        box.pack(fill="x", padx=8, pady=8)
        for bullet in bullets:
            ttk.Label(box, text=f"- {bullet}", wraplength=980, justify="left").pack(anchor="w")

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

    def _build_status_panel(self, parent: tk.Widget, tab_name: str) -> None:
        box = ttk.LabelFrame(parent, text=f"Status: {tab_name}", padding=8)
        box.pack(fill="both", padx=8, pady=8, expand=True)
        self.status_vars[tab_name] = tk.StringVar(value="Operacja: idle")
        self.page_status_vars[tab_name] = tk.StringVar(value="Strona: -")
        self.bbox_status_vars[tab_name] = tk.StringVar(value="BBox: -")
        self.count_status_vars[tab_name] = tk.StringVar(value="Liczniki: processed=0 accepted=0 skipped=0 errors=0")
        self.path_status_vars[tab_name] = tk.StringVar(value="Ścieżki: input=- output=-")
        self.stage_vars[tab_name] = tk.StringVar(value="Etap: idle")
        for var in (self.status_vars[tab_name], self.page_status_vars[tab_name], self.bbox_status_vars[tab_name], self.count_status_vars[tab_name], self.path_status_vars[tab_name], self.stage_vars[tab_name]):
            ttk.Label(box, textvariable=var).pack(anchor="w")
        text = tk.Text(box, height=8, state="disabled")
        text.pack(fill="both", expand=True)
        self.tab_logs[tab_name] = text

    def update_status_panel(self, tab_name: str, level: str, message: str) -> None:
        log = self.tab_logs.get(tab_name)
        if log is not None:
            log.configure(state="normal")
            log.insert("end", f"[{level}] {message}\n")
            log.see("end")
            log.configure(state="disabled")
        self.stage_vars.get(tab_name, tk.StringVar()).set(f"Etap: {message}")
        if "strona=" in message:
            self.page_status_vars[tab_name].set(f"Strona: {message}")
        if "bbox=" in message:
            self.bbox_status_vars[tab_name].set(f"BBox: {message}")

    def log_status(self, message: str) -> None:
        self.logger.info(message)
        self.after(0, lambda: self.update_status_panel("extract", "INFO", message))

    def log_error(self, message: str) -> None:
        self.logger.error(message)
        self.after(0, lambda: self.update_status_panel("extract", "ERROR", message))

    def _set_extract_running(self, running: bool) -> None:
        self.is_extracting = running
        self.extract_button.configure(state="disabled" if running else "normal")

    def _run_extract(self) -> None:
        if self.is_extracting:
            return

        pdf_path = self.pdf_path_var.get().strip()
        out_path = self.output_dir_var.get().strip()
        if not pdf_path:
            self.update_status_panel("extract", "ERROR", "Nie wybrano pliku PDF.")
            return
        if not out_path:
            self.update_status_panel("extract", "ERROR", "Nie wybrano katalogu output.")
            return

        self.source_pdf = Path(pdf_path)
        self.extract_output_dir = Path(out_path)

        self._set_extract_running(True)
        self.status_vars["extract"].set("Operacja: analyzing")
        self.update_status_panel("extract", "INFO", "Wczytywanie PDF")

        def worker() -> None:
            try:
                log_dir = ensure_log_dir(self.extract_output_dir)
                log_path = log_dir / "pdf_analyzer.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_path, encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
                root_logger = logging.getLogger()
                root_logger.addHandler(fh)

                self._save_ocr_settings()
                extractor = PDFExtractor(self.extract_output_dir, self.root_dir, self.settings, crop_padding_pt=float(self.crop_padding_var.get() or 10))
                records = extractor.extract(
                    self.source_pdf,
                    self.source_pdf.stem,
                    status_callback=lambda tab, level, msg: self.after(0, lambda: self.update_status_panel(tab, level, msg)),
                )
                page_count = len({r.page_number for r in records})
                box_count = len(records)
                self.after(0, lambda: self.status_vars["extract"].set(f"Operacja: Analyzed {page_count} pages, {box_count} boxes"))
                self.after(0, lambda: messagebox.showinfo("Done", "Extraction completed"))
                root_logger.removeHandler(fh)
                fh.close()
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self.update_status_panel("extract", "ERROR", str(exc)))
                self.after(0, lambda: self.status_vars["extract"].set("Operacja: failed"))
            finally:
                self.after(0, lambda: self._set_extract_running(False))

        threading.Thread(target=worker, daemon=True).start()


    def _select_tesseract(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Tesseract", "tesseract.exe"), ("Executable", "*.exe")])
        if p:
            self.tesseract_path_var.set(p)
            self._save_ocr_settings()

    def _save_ocr_settings(self) -> None:
        self.settings.tesseract_exe_path = self.tesseract_path_var.get().strip()
        self.settings.ocr_languages = self.ocr_lang_var.get().strip() or "eng+pol"
        save_settings(self.root_dir, self.settings)
        self.update_status_panel("extract", "INFO", "Zapisano ustawienia OCR")

    def _test_ocr(self) -> None:
        self._save_ocr_settings()
        runtime = OCRRuntime(self.root_dir, self.settings)
        status = runtime.diagnostic_test()
        self.ocr_status_var.set(f"OCR status: {status.message}")
        diagnostics = status.diagnostics or {}
        diag_txt = (
            f"tesseract_exe={diagnostics.get('tesseract_exe', 'n/a')} "
            f"tessdata={diagnostics.get('tessdata', 'n/a')}"
        )
        self.update_status_panel("extract", "INFO", f"OCR test: {status.state} path={status.tesseract_path} langs={status.languages} {diag_txt}")

    def _gen_dict(self) -> None:
        dict_dir = self.extract_output_dir / "dictionaries"
        dict_dir.mkdir(parents=True, exist_ok=True)
        SynonymBuilder(dict_dir / "master_dictionary.xlsx").ensure_dictionary_template()
        messagebox.showinfo("Done", "Dictionary template generated")

    def _select_dictionary(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if p:
            self.dictionary_path_var.set(p)

    def _freq(self) -> None:
        dict_path = Path(self.dictionary_path_var.get().strip())
        if not dict_path.exists():
            messagebox.showerror("Error", "Wskaż poprawny plik master_dictionary.xlsx")
            return
        sb = SynonymBuilder(dict_path)
        sb.analyze_text(self.extract_output_dir / "extraction_data.xlsx")
        messagebox.showinfo("Done", "Analyze Text completed")


    def _generate_synonym_candidates(self) -> None:
        dict_path = Path(self.dictionary_path_var.get().strip())
        if not dict_path.exists():
            messagebox.showerror("Error", "Wskaż poprawny plik master_dictionary.xlsx")
            return
        analyzed_path = self.extract_output_dir / "extraction_data.xlsx"
        if not analyzed_path.exists():
            messagebox.showerror("Error", "Brak pliku extraction_data.xlsx")
            return
        out_path = SynonymCandidateService().generate_candidates(
            analyzed_path,
            dict_path,
            self.extract_output_dir,
            status_callback=lambda tab, level, msg: self.update_status_panel(tab, level, msg),
        )
        df = pd.read_excel(out_path, sheet_name="Candidates")
        self.update_status_panel("classification", "INFO", f"Generated {len(df)} candidates: {out_path}")
        messagebox.showinfo("Done", f"Generated {len(df)} candidates\n{out_path}")

    def _import_accepted_synonyms(self) -> None:
        dict_path = Path(self.dictionary_path_var.get().strip())
        if not dict_path.exists():
            messagebox.showerror("Error", "Wskaż poprawny plik master_dictionary.xlsx")
            return

        default_path = self.extract_output_dir / "synonym_candidates.xlsx"
        candidates_path = default_path
        if not default_path.exists():
            selected = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
            if not selected:
                return
            candidates_path = Path(selected)

        imported = SynonymCandidateService().import_accepted_candidates(
            candidates_path,
            dict_path,
            confirm_overwrite=lambda msg: messagebox.askyesno("Confirm", msg),
            status_callback=lambda tab, level, msg: self.update_status_panel(tab, level, msg),
        )
        messagebox.showinfo("Done", f"Imported accepted synonyms: {imported}")

    def _classify(self) -> None:
        dict_path = Path(self.dictionary_path_var.get().strip())
        if not dict_path.exists():
            messagebox.showerror("Error", "Wskaż poprawny plik master_dictionary.xlsx")
            return
        DetailClassifier().classify(
            self.extract_output_dir / "extraction_data.xlsx",
            dict_path,
            self.extract_output_dir,
            status_callback=lambda tab, level, msg: self.update_status_panel(tab, level, msg),
        )
        self.update_status_panel("classification", "INFO", "Classification completed")
        messagebox.showinfo("Done", "Classification completed")

    def _export(self) -> None:
        export_dir = self.extract_output_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        Exporter().export(
            self.extract_output_dir / "extraction_data.xlsx",
            self.extract_output_dir / "classification_results.xlsx",
            export_dir,
            "PROJECT",
        )
        messagebox.showinfo("Done", "Metadata exported")
