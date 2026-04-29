# clean_curated.py
import shutil
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent
CURATED_DIR = PROJECT_ROOT / "data" / "curated"

print("=" * 60)
print("CLEANING CURATED DIRECTORY")
print("=" * 60)

if CURATED_DIR.exists():
    # Count files before deletion
    files = list(CURATED_DIR.glob("*.csv"))
    print(f"\n📁 Found {len(files)} files in {CURATED_DIR}")

    if len(files) > 0:
        # Ask for confirmation
        response = input(f"\n⚠️  Are you sure you want to delete all {len(files)} files? (yes/no): ")

        if response.lower() == 'yes':
            # Delete all CSV files
            for file in files:
                file.unlink()
                print(f"   🗑️  Deleted: {file.name}")

            print(f"\n✅ Deleted {len(files)} files from curated directory")
        else:
            print("\n❌ Operation cancelled")
    else:
        print("\n✅ Curated directory is already empty")
else:
    print(f"\n⚠️  Curated directory doesn't exist: {CURATED_DIR}")
    print("   Creating directory...")
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)