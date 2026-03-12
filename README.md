# Smart AI Data Warehouse

**University of Jordan – AI Department – Graduation Project 2026**  
**Supervisor:** Dr. Tamam Alsarhan  
**Team:** Layan Alkilani, Maha Qaddoumi, Laith Habash, Neveen Zabalawi  

## Overview
An automated data preprocessing system that combines a **rule‑based engine** with a **fine‑tuned FLAN‑T5 transformer** to generate optimal preprocessing plans for tabular datasets. The system produces AI‑ready datasets and logs every step into a SQLite registry.

## Repository Structure

├── configs/ # Configuration files (YAML / JSON)
├── data/ # All data (see subfolders)
│ ├── raw_input/ # Original datasets (CSV, Excel) – not tracked
│ ├── raw/ # Versioned raw copies (created by ingestion)
│ ├── curated/ # Cleaned data after transformation
│ ├── features/ # Final AI‑ready feature files
├── pipelines/ # Core pipeline modules (ingestion, validation, etc.)
├── models/ # Trained models (.pkl, transformer checkpoints)
├── registry/ # SQLite database + helper scripts
├── notebooks/ # (optional) Exploration / Colab notebooks
├── tests/ # Unit and integration tests
├── .gitignore
├── README.md
├── CONTRIBUTING.md
└── requirements.txt # Python dependencies


## Quick Start
1. Clone the repository  
   `git clone https://github.com/kilanilayan018-svg/Smart-AI-DataWarehouse.git`
2. Create a virtual environment  
   `python -m venv venv`  
   `source venv/bin/activate`  (or `venv\Scripts\activate` on Windows)
3. Install dependencies  
   `pip install -r requirements.txt`
4. Run the full pipeline on a sample dataset  
   `python main.py --input data/raw_input/sample.csv`

*(Detailed instructions will be added as the project progresses.)*

## Branching Strategy
We follow the **`main` / `develop` / `feature/*`** model:
- `main` – always production‑ready, stable code.
- `develop` – integration branch for completed features.
- `feature/*` – each new feature or task gets its own branch (e.g., `feature/ingestion-module`).  
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
This project is for educational purposes only.
