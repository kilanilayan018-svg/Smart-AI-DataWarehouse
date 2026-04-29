"""
Git Push Script - Smart AI Data Warehouse
Pushes today's pipeline changes to branch: website
Repo: https://github.com/kilanilayan018-svg/Smart-AI-DataWarehouse
"""

import subprocess
import sys
from datetime import datetime


REPO_URL = "https://github.com/kilanilayan018-svg/Smart-AI-DataWarehouse"
BRANCH   = "website"


def run(cmd: str, description: str):
    print(f"\n{'=' * 60}")
    print(f"▶ {description}")
    print(f"  $ {cmd}")
    print("=" * 60)

    result = subprocess.run(cmd, shell=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        print(f"   Exit code: {result.returncode}")
        sys.exit(result.returncode)

    print(f"✅ Done: {description}")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    commit_message = (
        f"pipeline updates {timestamp}: "
        f"fix _meta target leak in plan_generator, "
        f"fix pandas StringDtype in feature_engineering, "
        f"add schema target detection + value distribution, "
        f"add finetuning pair generator (T1.7), "
        f"update full pipeline runner with pair generation step"
    )

    print("=" * 60)
    print("GIT PUSH - SMART AI DATA WAREHOUSE")
    print(f"Branch:  {BRANCH}")
    print(f"Repo:    {REPO_URL}")
    print(f"Time:    {timestamp}")
    print("=" * 60)

    # ── 1. Make sure we're on the right branch ───────────────────
    run(
        f"git checkout {BRANCH}",
        f"Switch to branch '{BRANCH}'"
    )

    # ── 2. Pull latest to avoid conflicts ───────────────────────
    run(
        f"git pull origin {BRANCH}",
        "Pull latest from remote"
    )

    # ── 3. Stage all changes ─────────────────────────────────────
    run(
        "git add -A",
        "Stage all changes"
    )

    # ── 4. Show what's being committed ───────────────────────────
    print(f"\n{'=' * 60}")
    print("▶ Staged files summary:")
    print("=" * 60)
    subprocess.run("git status --short", shell=True, text=True)

    # ── 5. Commit ────────────────────────────────────────────────
    run(
        f'git commit -m "{commit_message}"',
        "Commit changes"
    )

    # ── 6. Push ──────────────────────────────────────────────────
    run(
        f"git push origin {BRANCH}",
        f"Push to origin/{BRANCH}"
    )

    print(f"\n{'=' * 60}")
    print("✅ ALL DONE — CODE PUSHED SUCCESSFULLY!")
    print("=" * 60)
    print(f"   Branch:  {BRANCH}")
    print(f"   Repo:    {REPO_URL}")
    print(f"   Commit:  {commit_message}")
    print("=" * 60)


if __name__ == "__main__":
    main()