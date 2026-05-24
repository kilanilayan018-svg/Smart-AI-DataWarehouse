from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------
# Folders / files to clean
# -----------------------------
CLEAN_TARGETS = [
    PROJECT_ROOT / "data" / "schema",
    PROJECT_ROOT / "data" / "curated",
    PROJECT_ROOT / "data" / "features",
    PROJECT_ROOT / "data" / "features_samples",
    PROJECT_ROOT / "data" / "finetuning",
    PROJECT_ROOT / "metadata" / "plans",
    PROJECT_ROOT / "logs" / "validation",
]

CLEAN_FILES = [
    PROJECT_ROOT / "logs" / "transformation_log.json",
]

# -----------------------------
# Keep placeholder files if present
# -----------------------------
KEEP_FILENAMES = {".gitkeep", ".gitignore"}


def safe_delete_contents(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for item in path.iterdir():
        if item.name in KEEP_FILENAMES:
            continue

        if item.is_file():
            item.unlink()

        elif item.is_dir():
            # Delete directory contents recursively
            for sub in item.rglob("*"):
                if sub.is_file():
                    sub.unlink()

            for sub in sorted(item.rglob("*"), reverse=True):
                if sub.is_dir():
                    try:
                        sub.rmdir()
                    except OSError:
                        pass

            try:
                item.rmdir()
            except OSError:
                pass


def clean_previous_outputs():
    print("=" * 70)
    print("CLEANING PREVIOUS GENERATED OUTPUTS")
    print("=" * 70)

    for folder in CLEAN_TARGETS:
        print(f"🧹 Cleaning folder: {folder}")
        safe_delete_contents(folder)

    for file_path in CLEAN_FILES:
        if file_path.exists():
            print(f"🗑️  Removing file: {file_path}")
            file_path.unlink()

    print("✅ Cleanup complete.\n")


def run_step(script_path: Path, label: str):
    print("=" * 70)
    print(f"RUNNING: {label}")
    print(f"SCRIPT:  {script_path}")
    print("=" * 70)

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    # If the script is inside pipelines/, run it as a module
    if script_path.parent.name == "pipelines":
        module_name = f"pipelines.{script_path.stem}"
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=PROJECT_ROOT,
            text=True
        )
    else:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            text=True
        )

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    print(f"✅ {label} completed.\n")


def main():
    print("=" * 70)
    print("SMART AI DATA WAREHOUSE - FULL PIPELINE RUNNER")
    print("=" * 70)
    print(f"📂 Project root: {PROJECT_ROOT}\n")

    clean_previous_outputs()

    steps = [
        ("Schema Extraction",          PROJECT_ROOT / "pipelines" / "schema_extractor.py"),
        ("Plan Dispatching",           PROJECT_ROOT / "pipelines" / "plan_dispatcher.py"),
        ("Validation",                 PROJECT_ROOT / "pipelines" / "run_validation.py"),
        ("Transformation",             PROJECT_ROOT / "pipelines" / "transformation_module.py"),
        ("Feature Engineering",        PROJECT_ROOT / "pipelines" / "feature_engineering.py"),

        # Legacy/training artifact generation.
        # The live decision plan is now created by plan_dispatcher.py above.
        ("Plan Generation",            PROJECT_ROOT / "pipelines" / "plan_generator.py"),
        ("Finetuning Pair Generation", PROJECT_ROOT / "pipelines" / "Pair_generator.py"),
    ]

    for label, script in steps:
        run_step(script, label)

    print("=" * 70)
    print("✅ FULL PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)
    print("Generated outputs:")
    print(f"  • Schema:            {PROJECT_ROOT / 'data' / 'schema'}")
    print(f"  • Validation logs:   {PROJECT_ROOT / 'logs' / 'validation'}")
    print(f"  • Curated data:      {PROJECT_ROOT / 'data' / 'curated'}")
    print(f"  • Features:          {PROJECT_ROOT / 'data' / 'features'}")
    print(f"  • Feature samples:   {PROJECT_ROOT / 'data' / 'features_samples'}")
    print(f"  • Plans:             {PROJECT_ROOT / 'metadata' / 'plans'}")
    print(f"  • Finetuning pairs:  {PROJECT_ROOT / 'data' / 'finetuning'}")
    print("=" * 70)


if __name__ == "__main__":
    main()