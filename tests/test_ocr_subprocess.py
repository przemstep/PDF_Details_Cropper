from pathlib import Path
import subprocess

from app.ocr_runtime import run_tesseract_ocr


def test_run_tesseract_ocr_success(monkeypatch, tmp_path: Path):
    exe = tmp_path / "tesseract.exe"
    exe.write_text("x")
    img = tmp_path / "a.png"
    img.write_text("x")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="hello", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = run_tesseract_ocr(img, exe)
    assert err is None
    assert text == "hello"


def test_run_tesseract_ocr_failure(monkeypatch, tmp_path: Path):
    exe = tmp_path / "tesseract.exe"
    exe.write_text("x")
    img = tmp_path / "a.png"
    img.write_text("x")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = run_tesseract_ocr(img, exe)
    assert text == ""
    assert "OCR execution failed" in err


def test_run_tesseract_ocr_timeout(monkeypatch, tmp_path: Path):
    exe = tmp_path / "tesseract.exe"
    exe.write_text("x")
    img = tmp_path / "a.png"
    img.write_text("x")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = run_tesseract_ocr(img, exe, timeout_seconds=1)
    assert text == ""
    assert "timeout" in err


def test_run_tesseract_ocr_missing_executable(tmp_path: Path):
    img = tmp_path / "a.png"
    img.write_text("x")
    text, err = run_tesseract_ocr(img, tmp_path / "missing.exe")
    assert text == ""
    assert "missing executable" in err
