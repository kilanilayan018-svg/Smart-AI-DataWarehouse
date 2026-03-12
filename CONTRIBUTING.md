# Contributing to Smart AI Data Warehouse

We’re glad you want to contribute! Please follow these guidelines to keep the repository clean and maintainable.

## Branching Model
- **`main`** – Always reflects the latest stable, production‑ready code. Only maintainers merge into `main`.
- **`develop`** – The main integration branch. All completed features are merged here. It should always be in a working state.
- **Feature branches** – Named `feature/short-description` (e.g., `feature/ingestion-module`).  
  Create them from the latest `develop` and merge back via Pull Request.

## Workflow
1. **Sync your local `develop`**  
   ```bash
   git checkout develop
   git pull origin develop
