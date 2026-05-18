# PDF Details Cropper

Desktopowa aplikacja Tkinter do półautomatycznej ekstrakcji detali z annotation boxów PDF, budowy trwałego słownika synonimów, klasyfikacji technicznej oraz eksportu plików i metadanych pod SharePoint.

## Workflow
1. **Extract**: odczyt annotation boxów, crop PDF/PNG, ekstrakcja tekstu do `intermediate/`.
2. **Dictionary Builder**: analiza częstości fraz 1-3 wyrazowych i utrzymanie `data/dictionaries/master_dictionary.xlsx`.
3. **Classification**: scoring na bazie słownika, priorytetów, negative synonyms + generacja keywords/search tags.
4. **Export**: finalne metadane i nazewnictwo bezpieczne dla SharePoint.

## Uruchomienie
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python main.py
```

## Build .exe (manual)
Repo zawiera GitHub Action `Build Windows EXE` uruchamianą ręcznie przez **workflow_dispatch**.
Po uruchomieniu pobierz artifact `PDF_Details_Cropper-exe`.

Przed buildem upewnij się, że zależności zawierają `pytesseract` i `Pillow` (np. przez `pip install -r requirements.txt`).
