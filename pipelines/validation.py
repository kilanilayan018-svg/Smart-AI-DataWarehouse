import pandas as pd


class DataValidationModule:

    def __init__(self, df, schema, target_column=None):
        self.df = df
        self.schema = schema
        self.target_column = target_column
        self.errors = []

    def check_columns(self):
        dataset_columns = set(self.df.columns)
        schema_columns = set(self.schema.keys())

        missing = schema_columns - dataset_columns
        extra = dataset_columns - schema_columns

        if missing:
            self.errors.append(f"Missing columns: {missing}")

        if extra:
            self.errors.append(f"Extra columns: {extra}")

    def check_target(self):
        if self.target_column is None:
            return

        if self.target_column not in self.df.columns:
            self.errors.append(f"Target column missing: {self.target_column}")

    def check_types(self):
        for col in self.schema:
            if col not in self.df.columns:
                continue

            expected_type = self.schema[col]["dtype"]
            actual_type = str(self.df[col].dtype)

            if expected_type != actual_type:
                self.errors.append(f"Type mismatch in {col}")

    def check_missing(self):
        for col in self.schema:
            if col not in self.df.columns:
                continue

            expected_missing = self.schema[col]["missing_count"]
            actual_missing = self.df[col].isnull().sum()

            if expected_missing != actual_missing:
                self.errors.append(f"Missing mismatch in {col}")

    def validate(self):
        self.errors = []

        self.check_columns()
        self.check_target()
        self.check_types()
        self.check_missing()

        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors
        }