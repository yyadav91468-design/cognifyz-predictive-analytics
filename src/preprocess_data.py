
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "daily_sales_features.csv"
SUMMARY_FILE = PROJECT_ROOT / "reports" / "data_quality_report.txt"


def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """Load transactions and remove unusable sales records."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Download Online Retail.xlsx and place it in data/raw/."
        )

    print("Loading retail dataset...")
    df = pd.read_excel(file_path, engine="openpyxl")

    original_rows = len(df)

    required_columns = [
        "InvoiceNo",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    invalid_dates = int(df["InvoiceDate"].isna().sum())

    df = df.dropna(
        subset=["InvoiceDate", "Quantity", "UnitPrice", "InvoiceNo"]
    ).copy()

    # Cancellation invoices are commonly marked with a C prefix.
    cancellation_mask = (
        df["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    )
    cancellation_rows = int(cancellation_mask.sum())
    df = df.loc[~cancellation_mask].copy()

    # Exclude returns, non-positive quantities, and invalid prices
    # from this positive-sales forecasting baseline.
    invalid_sales_mask = (
        (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
    )
    invalid_sales_rows = int(invalid_sales_mask.sum())
    df = df.loc[~invalid_sales_mask].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.normalize()

    print(f"Original transaction rows: {original_rows:,}")
    print(f"Invalid dates: {invalid_dates:,}")
    print(f"Cancellation rows excluded: {cancellation_rows:,}")
    print(f"Other invalid sales rows excluded: {invalid_sales_rows:,}")
    print(f"Valid positive-sales rows: {len(df):,}")

    if df.empty:
        raise ValueError("No valid sales records remain after cleaning.")

    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by date and build next-day prediction features."""

    daily = (
        df.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .rename(columns={"Revenue": "daily_sales"})
        .sort_values("Date")
        .set_index("Date")
    )

    # Add dates without recorded transactions as zero-sales dates.
    full_dates = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D",
    )
    daily = daily.reindex(full_dates, fill_value=0.0)
    daily.index.name = "Date"

    # Calendar features
    daily["day_of_week"] = daily.index.dayofweek
    daily["day_of_month"] = daily.index.day
    daily["month"] = daily.index.month
    daily["day_of_year"] = daily.index.dayofyear
    daily["is_weekend"] = (daily.index.dayofweek >= 5).astype(int)

    # Lag features use only sales already observed on or before the
    # feature date. Rolling averages are shifted to prevent leakage.
    for lag in (0, 1, 2, 7, 14, 28):
        daily[f"sales_lag_{lag}"] = daily["daily_sales"].shift(lag)

    daily["rolling_mean_7"] = (
        daily["daily_sales"].shift(1).rolling(window=7).mean()
    )
    daily["rolling_mean_14"] = (
        daily["daily_sales"].shift(1).rolling(window=14).mean()
    )
    daily["rolling_std_7"] = (
        daily["daily_sales"].shift(1).rolling(window=7).std()
    )

    # Target: revenue on the following calendar day.
    daily["target_next_day_sales"] = daily["daily_sales"].shift(-1)

    # Initial rows lack sufficient history; final row lacks a next-day label.
    daily = daily.dropna().reset_index()

    # The current day's sales are retained in the raw daily series for
    # reporting, but excluded as a direct model input.
    feature_columns = [
        "Date",
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year",
        "is_weekend",
        "sales_lag_0",
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "target_next_day_sales",
    ]

    return daily[feature_columns]


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    transactions = load_and_clean_data(RAW_FILE)
    features = build_daily_features(transactions)

    features.to_csv(OUTPUT_FILE, index=False)

    report = [
        "COGNEVANCE PREDICTIVE ANALYTICS - DATA QUALITY REPORT",
        "=" * 58,
        f"Source file: {RAW_FILE.name}",
        f"Valid transaction records: {len(transactions):,}",
        f"Unique transaction dates: {transactions['Date'].nunique():,}",
        f"Date range: {transactions['Date'].min().date()} to "
        f"{transactions['Date'].max().date()}",
        f"Prepared daily feature rows: {len(features):,}",
        f"Feature columns: {len(features.columns) - 2}",
        f"Missing values in prepared dataset: {int(features.isna().sum().sum())}",
        f"Output file: {OUTPUT_FILE}",
        "",
        "Target: next calendar day's positive sales revenue in GBP.",
        "Cancellations, returns, and non-positive sales are excluded.",
        "This is a forecasting baseline, not a guarantee of future performance.",
        "Dataset: UCI Online Retail, DOI: 10.24432/C5BW33",
        "License: CC BY 4.0",
    ]

    SUMMARY_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\nPreprocessing completed successfully.")
    print(f"Prepared dataset: {OUTPUT_FILE}")
    print(f"Quality report: {SUMMARY_FILE}")
    print(f"Feature dataset shape: {features.shape}")
    print(features.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
