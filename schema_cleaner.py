import os
from pathlib import Path

SCHEMA_DIR = "data/schema/"

def clear_schemas(schema_dir=SCHEMA_DIR):
    schema_path = Path(schema_dir)

    if not schema_path.exists():
        print(f"⚠️  Schema folder not found: {schema_dir}")
        return

    json_files = list(schema_path.glob("*.json"))

    if not json_files:
        print("✅ Schema folder is already empty, nothing to delete.")
        return

    print(f"🗑️  Found {len(json_files)} schema file(s) to delete:\n")

    deleted = 0
    for f in json_files:
        try:
            f.unlink()
            print(f"   ❌ Deleted: {f.name}")
            deleted += 1
        except Exception as e:
            print(f"   ⚠️  Could not delete {f.name}: {e}")

    print(f"\n✅ Done! Deleted {deleted}/{len(json_files)} schema files.")
    print(f"📁 Folder is clean: {schema_dir}")


if __name__ == "__main__":
    print("=" * 50)
    print("SCHEMA CLEANER")
    print("=" * 50 + "\n")
    clear_schemas()
    print("\n" + "=" * 50)