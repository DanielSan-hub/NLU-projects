from __future__ import annotations

import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def svg_size(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    width = int(float(root.attrib.get("width", "1200")))
    height = int(float(root.attrib.get("height", "760")))
    return width, height


def main() -> None:
    browser = next((p for p in CHROME_CANDIDATES if p.exists()), None)
    if browser is None:
        raise SystemExit("No Chrome/Edge executable found for conversion.")
    png_dir = FIG_DIR / "png"
    pdf_dir = FIG_DIR / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for svg in sorted(FIG_DIR.glob("*.svg")):
        width, height = svg_size(svg)
        png_out = png_dir / f"{svg.stem}.png"
        pdf_out = pdf_dir / f"{svg.stem}.pdf"
        url = file_uri(svg)
        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={width},{height}",
                f"--screenshot={png_out}",
                url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_out}",
                url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )

    final_fig_dir = WORKSPACE_ROOT / "NLU_results" / "extracted_final_results" / "reports" / "figures"
    if final_fig_dir.exists():
        for sub in ["png", "pdf"]:
            dest = final_fig_dir / sub
            dest.mkdir(parents=True, exist_ok=True)
            for path in (FIG_DIR / sub).glob("*"):
                target = dest / path.name
                target.write_bytes(path.read_bytes())
    print(f"Converted SVG figures to PNG/PDF using {browser}")


if __name__ == "__main__":
    main()
