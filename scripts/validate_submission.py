import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS = ["LM/partA", "LM/partB", "NLU/partA", "NLU/partB"]
REQUIRED_FLAGS = [
    "--mode",
    "--device",
    "--seed",
    "--resume",
    "--allow-cpu",
    "--amp",
    "--log-tensorboard",
    "--num-workers",
    "--pin-memory",
]
REQUIRED_SCRIPTS = [
    "scripts/smoke_all.sh",
    "scripts/run_core.sh",
    "scripts/run_extras.sh",
    "scripts/collect_results.py",
    "scripts/validate_submission.py",
]
RESULT_CSV = {
    "LM/partA": "results_partA.csv",
    "LM/partB": "results_partB.csv",
    "NLU/partA": "results_partA.csv",
    "NLU/partB": "results_partB.csv",
}
REPORT_FILES = {
    "LM/partA": "reports/LM_partA_report.md",
    "LM/partB": "reports/LM_partB_report.md",
    "NLU/partA": "reports/NLU_partA_report.md",
    "NLU/partB": "reports/NLU_partB_report.md",
}


class Checklist:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def pass_item(self, message: str) -> None:
        print(f"[PASS] {message}")

    def warn_item(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail_item(self, message: str) -> None:
        self.errors.append(message)
        print(f"[FAIL] {message}")

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.pass_item(message)
        else:
            self.fail_item(message)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def csv_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return bool(rows)


def has_file_named(root: Path, names: list[str]) -> bool:
    return any((root / name).exists() or any(root.rglob(name)) for name in names)


def check_part(part: str, checklist: Checklist) -> None:
    part_dir = ROOT / part
    checklist.require(part_dir.is_dir(), f"{part}/ exists")
    if not part_dir.exists():
        return

    for filename in ["main.py", "model.py", "functions.py", "utils.py", "README.md"]:
        checklist.require((part_dir / filename).is_file(), f"{part}/{filename} exists")

    for dirname in ["dataset", "bin", "results"]:
        checklist.require((part_dir / dirname).is_dir(), f"{part}/{dirname}/ exists")

    main_path = part_dir / "main.py"
    if main_path.exists():
        text = main_path.read_text(encoding="utf-8")
        for flag in REQUIRED_FLAGS:
            checklist.require(flag in text, f"{part}/main.py supports {flag}")
        absolute_root = str(ROOT)
        checklist.require(
            absolute_root not in text and absolute_root.replace("\\", "\\\\") not in text,
            f"{part}/main.py has no absolute local project path",
        )

    notebooks = list(part_dir.rglob("*.ipynb"))
    checklist.require(not notebooks, f"{part}/ contains no notebooks")

    result_csv = part_dir / "results" / RESULT_CSV[part]
    checklist.require(result_csv.is_file(), f"{relative(result_csv)} exists")
    if result_csv.exists():
        if csv_has_data(result_csv):
            checklist.pass_item(f"{relative(result_csv)} has result rows")
        else:
            checklist.warn_item(f"{relative(result_csv)} exists but has no result rows")

    report_path = ROOT / REPORT_FILES[part]
    checklist.require(report_path.is_file(), f"{relative(report_path)} exists")

    latest = part_dir / "results" / "latest_run.txt"
    checklist.require(latest.is_file(), f"{part}/results/latest_run.txt exists")
    if not latest.exists():
        return

    run_name = latest.read_text(encoding="utf-8").strip()
    run_dir = part_dir / "results" / run_name
    checklist.require(run_dir.is_dir(), f"{relative(run_dir)} exists")
    if not run_dir.exists():
        return

    checklist.require(has_file_named(run_dir, ["config.json"]), f"{part} latest run has config.json")
    checklist.require(has_file_named(run_dir, ["summary.txt"]), f"{part} latest run has summary.txt")

    if part == "LM/partB":
        checklist.require(
            has_file_named(run_dir, ["best_lora_adapters.pt"]),
            f"{part} latest run has best_lora_adapters.pt",
        )
        checklist.require(
            has_file_named(run_dir, ["last_lora_adapters.pt"]),
            f"{part} latest run has last_lora_adapters.pt",
        )
    elif part == "NLU/partB":
        experiment_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
        checklist.require(bool(experiment_dirs), f"{part} latest run has model experiment directories")
        for experiment_dir in experiment_dirs:
            label = experiment_dir.name
            checklist.require((experiment_dir / "best.pt").is_file(), f"{part} {label} best.pt exists")
            checklist.require((experiment_dir / "last.pt").is_file(), f"{part} {label} last.pt exists")
    else:
        checklist.require(has_file_named(run_dir, ["best.pt"]), f"{part} latest run has best.pt")
        checklist.require(has_file_named(run_dir, ["last.pt"]), f"{part} latest run has last.pt")


def main() -> None:
    checklist = Checklist()

    for script in REQUIRED_SCRIPTS:
        checklist.require((ROOT / script).is_file(), f"{script} exists")

    for dirname in ["logs", "results", "reports"]:
        checklist.require((ROOT / dirname).is_dir(), f"root {dirname}/ exists")

    for part in PARTS:
        print(f"\n== {part} ==")
        check_part(part, checklist)

    print("\n== Summary ==")
    if checklist.warnings:
        print(f"Warnings: {len(checklist.warnings)}")
    else:
        print("Warnings: 0")
    if checklist.errors:
        print(f"Errors: {len(checklist.errors)}")
        raise SystemExit(1)
    print("Validation passed: required structure, scripts, results, checkpoints, and reports are present.")


if __name__ == "__main__":
    main()
