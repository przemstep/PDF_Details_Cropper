from pathlib import Path

from app.gui import AppGUI
from app.utils import setup_logging


def main() -> None:
    root = Path(__file__).parent
    setup_logging(root / "logs/app.log")
    app = AppGUI(root)
    app.mainloop()


if __name__ == "__main__":
    main()
