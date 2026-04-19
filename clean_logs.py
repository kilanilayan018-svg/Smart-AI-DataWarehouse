# clean_logs.py
import shutil
from pathlib import Path
import json

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
TRANSFORM_LOG_DIR = LOG_DIR / "transform"
MASTER_LOG_FILE = LOG_DIR / "transformation_log.json"


def delete_transform_logs():
    """Delete all transformation log files"""
    print("=" * 60)
    print("CLEANING TRANSFORMATION LOGS")
    print("=" * 60)

    deleted_count = 0
    failed_count = 0

    # Delete individual transform logs
    if TRANSFORM_LOG_DIR.exists():
        log_files = list(TRANSFORM_LOG_DIR.glob("*.json"))

        if log_files:
            print(f"\n📁 Found {len(log_files)} log files in {TRANSFORM_LOG_DIR}")

            for log_file in log_files:
                try:
                    log_file.unlink()
                    print(f"   🗑️  Deleted: {log_file.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to delete {log_file.name}: {e}")
                    failed_count += 1
        else:
            print(f"\n✅ No log files found in {TRANSFORM_LOG_DIR}")
    else:
        print(f"\n⚠️  Transform log directory doesn't exist: {TRANSFORM_LOG_DIR}")
        print(f"   Creating directory...")
        TRANSFORM_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Delete master log file if it exists
    if MASTER_LOG_FILE.exists():
        try:
            MASTER_LOG_FILE.unlink()
            print(f"\n🗑️  Deleted master log: {MASTER_LOG_FILE.name}")
            deleted_count += 1
        except Exception as e:
            print(f"\n❌ Failed to delete master log: {e}")
            failed_count += 1
    else:
        print(f"\n✅ No master log file found")

    # Summary
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE!")
    print("=" * 60)
    print(f"   ✅ Successfully deleted: {deleted_count} files")
    if failed_count > 0:
        print(f"   ❌ Failed to delete: {failed_count} files")
    print("=" * 60)

    return deleted_count


def preview_logs():
    """Preview what will be deleted without actually deleting"""
    print("=" * 60)
    print("PREVIEW: LOGS TO BE DELETED")
    print("=" * 60)

    files_to_delete = []

    if TRANSFORM_LOG_DIR.exists():
        for log_file in TRANSFORM_LOG_DIR.glob("*.json"):
            size_kb = log_file.stat().st_size / 1024
            files_to_delete.append({
                "path": log_file,
                "name": log_file.name,
                "size_kb": size_kb
            })

    if MASTER_LOG_FILE.exists():
        size_kb = MASTER_LOG_FILE.stat().st_size / 1024
        files_to_delete.append({
            "path": MASTER_LOG_FILE,
            "name": MASTER_LOG_FILE.name,
            "size_kb": size_kb
        })

    if files_to_delete:
        print(f"\n📁 Found {len(files_to_delete)} log file(s) to delete:\n")
        for f in files_to_delete:
            print(f"   - {f['name']} ({f['size_kb']:.1f} KB)")

        total_size = sum(f['size_kb'] for f in files_to_delete)
        print(f"\n   Total size: {total_size:.1f} KB")
    else:
        print("\n✅ No log files found to delete")

    print("=" * 60)
    return files_to_delete


def interactive_clean():
    """Interactive cleanup with confirmation"""
    print("=" * 60)
    print("INTERACTIVE LOG CLEANUP")
    print("=" * 60)

    # Preview what will be deleted
    files = preview_logs()

    if not files:
        print("\n❌ No files to delete. Exiting.")
        return

    # Ask for confirmation
    print("\n⚠️  This will permanently delete all transformation log files!")
    response = input("\nAre you sure you want to continue? (yes/no): ")

    if response.lower() == 'yes':
        print("\nProceeding with deletion...\n")
        delete_transform_logs()
    else:
        print("\n❌ Operation cancelled. No files were deleted.")


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--preview":
            preview_logs()
        elif sys.argv[1] == "--force":
            delete_transform_logs()
        elif sys.argv[1] == "--help":
            print("""
Usage: python clean_logs.py [OPTION]

Options:
  --preview    Preview logs that would be deleted without deleting
  --force      Delete all logs without confirmation
  --help       Show this help message

Without options, runs in interactive mode with confirmation.
            """)
        else:
            interactive_clean()
    else:
        interactive_clean()