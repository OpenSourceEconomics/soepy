import json
import pickle
from pathlib import Path

import pandas as pd

TEST_RESOURCES_DIR = Path(__file__).parent.absolute()


def load_pandas_object(filepath, metadata_filepath):
    """Load pandas DataFrame or Series from CSV using metadata"""
    # Load metadata
    with open(metadata_filepath) as f:
        metadata = json.load(f)

    # Determine how many index columns to read
    index_cols = (
        list(range(metadata["index_nlevels"]))
        if metadata["index_nlevels"] > 0
        else None
    )

    # Read CSV
    if metadata["column_nlevels"] and metadata["column_nlevels"] > 1:
        # Multi-level columns
        header = list(range(metadata["column_nlevels"]))
        df = pd.read_csv(filepath, index_col=index_cols, header=header)
    else:
        df = pd.read_csv(filepath, index_col=index_cols)

    # Restore index names
    if metadata["index_names"]:
        if metadata["index_nlevels"] == 1:
            df.index.name = metadata["index_names"][0]
        else:
            df.index.names = metadata["index_names"]

    # Restore column names if DataFrame
    if not metadata["is_series"] and metadata["column_names"]:
        if metadata["column_nlevels"] == 1:
            df.columns.name = metadata["column_names"][0]
        else:
            df.columns.names = metadata["column_names"]

    # Convert back to Series if needed
    if metadata["is_series"]:
        series = df.iloc[:, 0]
        series.name = metadata.get("series_name")
        return series

    return df


def import_vault_from_stable_format():
    """Import vault from stable CSV/pickle format to new pickle format"""

    input_dir = TEST_RESOURCES_DIR / "vault_migration"
    vault_file = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"

    # Count how many test cases exist
    test_dirs = sorted(input_dir.glob("test_*"))
    num_tests = len(test_dirs)

    print(f"Found {num_tests} test cases to import")

    vault = {}

    for i in range(num_tests):
        print(f"Importing test case {i}...")

        test_dir = input_dir / f"test_{i:03d}"

        # Load dict from pickle
        with open(test_dir / "model_spec_init_dict.pkl", "rb") as f:
            model_spec_init_dict = pickle.load(f)

        # Load all DataFrames/Series from CSV
        random_model_params_df = load_pandas_object(
            test_dir / "random_model_params_df.csv",
            test_dir / "random_model_params_df.json",
        )
        exog_educ_shares = load_pandas_object(
            test_dir / "exog_educ_shares.csv", test_dir / "exog_educ_shares.json"
        )
        exog_child_age_shares = load_pandas_object(
            test_dir / "exog_child_age_shares.csv",
            test_dir / "exog_child_age_shares.json",
        )
        exog_partner_shares = load_pandas_object(
            test_dir / "exog_partner_shares.csv", test_dir / "exog_partner_shares.json"
        )
        exog_exper_shares_pt = load_pandas_object(
            test_dir / "exog_exper_shares_pt.csv",
            test_dir / "exog_exper_shares_pt.json",
        )
        exog_exper_shares_ft = load_pandas_object(
            test_dir / "exog_exper_shares_ft.csv",
            test_dir / "exog_exper_shares_ft.json",
        )
        exog_child_info = load_pandas_object(
            test_dir / "exog_child_info.csv", test_dir / "exog_child_info.json"
        )
        exog_partner_arrival_info = load_pandas_object(
            test_dir / "exog_partner_arrival_info.csv",
            test_dir / "exog_partner_arrival_info.json",
        )
        exog_partner_separation_info = load_pandas_object(
            test_dir / "exog_partner_separation_info.csv",
            test_dir / "exog_partner_separation_info.json",
        )
        expected_df = load_pandas_object(
            test_dir / "expected_df.csv", test_dir / "expected_df.json"
        )
        expected_df_unbiased = load_pandas_object(
            test_dir / "expected_df_unbiased.csv",
            test_dir / "expected_df_unbiased.json",
        )

        vault[i] = (
            model_spec_init_dict,
            random_model_params_df,
            exog_educ_shares,
            exog_child_age_shares,
            exog_partner_shares,
            exog_exper_shares_pt,
            exog_exper_shares_ft,
            exog_child_info,
            exog_partner_arrival_info,
            exog_partner_separation_info,
            expected_df,
            expected_df_unbiased,
        )

    print(f"\nSaving new vault to {vault_file}...")
    with open(vault_file, "wb") as file:
        pickle.dump(vault, file)

    print("✓ Import complete!")
    print(f"  Imported {len(vault)} test cases")


if __name__ == "__main__":
    import_vault_from_stable_format()
