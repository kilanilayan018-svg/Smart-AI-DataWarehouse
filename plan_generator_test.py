"""
Run PlanGenerator on all schema files
Location: project root
Usage: python run_plan_generator.py
"""

from pathlib import Path
from pipelines.plan_generator import PlanGenerator, load_schema


def main():
    print("\n" + "=" * 60)
    print("RUN PLAN GENERATOR (ROOT TEST)")
    print("=" * 60 + "\n")

    schema_folder = Path("data/schema")
    output_folder = Path("metadata/plans")
    output_folder.mkdir(parents=True, exist_ok=True)

    if not schema_folder.exists():
        print("❌ data/schema folder not found")
        return

    schema_files = list(schema_folder.glob("*_schema.json"))

    if not schema_files:
        print("❌ No schema files found")
        return

    print(f"📁 Found {len(schema_files)} schema files\n")

    generator = PlanGenerator()

    for i, schema_file in enumerate(schema_files, 1):
        print(f"{i}. Processing: {schema_file.name}")

        try:
            dataset_name = schema_file.stem.replace("_schema", "")
            schema = load_schema(schema_file)

            plan = generator.generate_plan(schema, dataset_name)
            saved_path = generator.save_plan(plan, dataset_name)

            # 🔥 PRINT CLEAN OUTPUT
            print(f"   📌 Dataset: {dataset_name}")
            print(f"   🎯 Target: {plan['target_column']}")
            print(f"   🧠 Task: {plan['task_type']}")
            print(f"   🧹 Drop: {plan['preprocessing']['drop_columns']}")
            print(f"   ⚙️ Scale: {len(plan['preprocessing']['scaling']['columns'])} columns")
            print(f"   🔄 Encoding (one-hot): {plan['preprocessing']['encoding']['one_hot_columns']}")
            print(f"   🔄 Encoding (label): {plan['preprocessing']['encoding']['label_encoding_columns']}")
            print(f"   💾 Saved → {saved_path}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("=" * 60)
    print("✅ ALL PLANS GENERATED")
    print("📁 Check metadata/plans/")
    print("=" * 60)


if __name__ == "__main__":
    main()