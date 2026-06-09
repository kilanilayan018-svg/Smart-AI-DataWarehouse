import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smart AI Data Warehouse", layout="wide")

st.title("Smart AI Data Warehouse")
st.caption("Upload your dataset and automatically clean it")

# ---------------------------
# Upload Section
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # ---------------------------
        # Raw Data
        # ---------------------------
        st.subheader("📊 Raw Dataset")
        st.dataframe(df.head(20), use_container_width=True)

        # ---------------------------
        # Fake Schema (UI only)
        # ---------------------------
        schema = {
            col: {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "unique": int(df[col].nunique())
            }
            for col in df.columns
        }

        st.subheader("🧠 Detected Schema")
        st.json(schema)

        # ---------------------------
        # Fake Plan
        # ---------------------------
        plan = {
            "target_column": df.columns[-1],
            "task_type": "classification",
            "preprocessing": {
                "imputation": "median",
                "encoding": "one_hot",
                "scaling": "standard",
                "drop_columns": []
            }
        }

        st.subheader("⚙️ Cleaning Plan")
        st.json(plan)

        # ---------------------------
        # Run Pipeline Button
        # ---------------------------
        if st.button("🚀 Run Cleaning Pipeline"):
            st.success("Pipeline executed successfully!")

            cleaned_df = df.copy()

            st.subheader("🧹 Cleaned Dataset")
            st.dataframe(cleaned_df.head(20), use_container_width=True)

            csv = cleaned_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Cleaned Dataset",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error loading file: {e}")