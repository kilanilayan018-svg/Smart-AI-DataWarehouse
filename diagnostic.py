from pathlib import Path

curated = Path("data/curated")

print("ALL files in curated:")
for f in sorted(curated.glob("*.csv")):
    print(f"  - {f.name}")

print("\nFiles matched by get_cleaned_files():")
for f in sorted(curated.glob("*_cleaned.csv")):
    print(f"  - {f.name}")