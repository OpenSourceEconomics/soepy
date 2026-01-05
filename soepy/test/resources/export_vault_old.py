import json
import os
import pickle
from pathlib import Path

import pandas as pd

TEST_RESOURCES_DIR = Path(__file__).parent.absolute()


def save_pandas_object(obj, filepath, metadata_filepath):
    """Save pandas DataFrame or Series to CSV with metadata"""
    # Collect metadata about the object
    metadata = {
        "is_series": isinstance(obj, pd.Series),
        "index_names": None,
        "index_nlevels": None,
        "column_names": None,
        "column_nlevels": None,
    }

    if isinstance(obj, pd.Series):
        df = obj.to_frame()
        metadata["series_name"] = obj.name
        metadata["index_names"] = [name for name in df.index.names]
        metadata["index_nlevels"] = df.index.nlevels
    else:
        df = obj
        metadata["index_names"] = [name for name in df.index.names]
        metadata["index_nlevels"] = df.index.nlevels
        metadata["column_names"] = [name for name in df.columns.names]
        metadata["column_nlevels"] = df.columns.nlevels

    # Save metadata
    with open(metadata_filepath, "w") as f:
        json.dump(metadata, f)

    # Save CSV
    df.to_csv(filepath)


def export_vault_to_stable_format():
    """Export vault from old pickle format to stable CSV/pickle format"""

    vault_file = TEST_RESOURCES_DIR / "regression_vault.soepy.pkl"
    output_dir = TEST_RESOURCES_DIR / "vault_migration"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading old vault...")
    with open(vault_file, "rb") as file:
        tests_sim_func = pickle.load(file)

    print(f"Found {len(tests_sim_func)} test cases")

    # Export each test case
    for i in range(len(tests_sim_func)):
        print(f"Exporting test case {i}...")

        test_dir = output_dir / f"test_{i:03d}"
        test_dir.mkdir(exist_ok=True)

        (
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
        ) = tests_sim_func[i]

        # Save dict as pickle with protocol 2 (more stable)
        with open(test_dir / "model_spec_init_dict.pkl", "wb") as f:
            pickle.dump(model_spec_init_dict, f, protocol=2)

        # Save all DataFrames/Series as CSV with metadata
        save_pandas_object(
            random_model_params_df,
            test_dir / "random_model_params_df.csv",
            test_dir / "random_model_params_df.json",
        )
        save_pandas_object(
            exog_educ_shares,
            test_dir / "exog_educ_shares.csv",
            test_dir / "exog_educ_shares.json",
        )
        save_pandas_object(
            exog_child_age_shares,
            test_dir / "exog_child_age_shares.csv",
            test_dir / "exog_child_age_shares.json",
        )
        save_pandas_object(
            exog_partner_shares,
            test_dir / "exog_partner_shares.csv",
            test_dir / "exog_partner_shares.json",
        )
        save_pandas_object(
            exog_exper_shares_pt,
            test_dir / "exog_exper_shares_pt.csv",
            test_dir / "exog_exper_shares_pt.json",
        )
        save_pandas_object(
            exog_exper_shares_ft,
            test_dir / "exog_exper_shares_ft.csv",
            test_dir / "exog_exper_shares_ft.json",
        )
        save_pandas_object(
            exog_child_info,
            test_dir / "exog_child_info.csv",
            test_dir / "exog_child_info.json",
        )
        save_pandas_object(
            exog_partner_arrival_info,
            test_dir / "exog_partner_arrival_info.csv",
            test_dir / "exog_partner_arrival_info.json",
        )
        save_pandas_object(
            exog_partner_separation_info,
            test_dir / "exog_partner_separation_info.csv",
            test_dir / "exog_partner_separation_info.json",
        )
        save_pandas_object(
            expected_df, test_dir / "expected_df.csv", test_dir / "expected_df.json"
        )
        save_pandas_object(
            expected_df_unbiased,
            test_dir / "expected_df_unbiased.csv",
            test_dir / "expected_df_unbiased.json",
        )

    print(f"\n✓ Export complete! Data saved to {output_dir}")
    print(f"  Total test cases: {len(tests_sim_func)}")


if __name__ == "__main__":
    export_vault_to_stable_format()
