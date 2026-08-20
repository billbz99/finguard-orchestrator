"""
SWIFT Log Generator Utility
Transforms tabular SAML-D transaction ledgers into realistic SWIFT ISO 20022/MT103 
unstructured text blocks rendered for downstream semantic chunking and vector ingestion.
"""

from pathlib import Path
import pandas as pd


def format_swift_block(row: dict) -> str:
    """Formats a single transaction row into a standardized SWIFT MT103 block."""
    date_str = str(row["Date"]).replace("-", "")[2:]  # Convert YYYY-MM-DD to YYMMDD
    currency = "USD" if row["Payment_currency"] == "US Dollar" else "GBP"
    amount_str = f"{float(row['Amount']):.2f}".replace(".", ",")

    sender_loc = row["Sender_bank_location"]
    receiver_loc = row["Receiver_bank_location"]

    swift_msg = f"""
{'{'}1:F01BANK{sender_loc}22AXXX0000000000{'}'}
{'{'}2:I103BANK{receiver_loc}22XXXXN{'}'}
{'{'}4:
:20:TXN-{date_str}-{row['Sender_account']}
:32A:{date_str}{currency}{amount_str}
:50K:/{row['Sender_account']}
SENDER LOCATION: {sender_loc}
:59:/{row['Receiver_account']}
BENEFICIARY LOCATION: {receiver_loc}
:70:TYPE: {row['Payment_type']} | MEMO: {row['Laundering_type']}
-{'}'}
"""
    return swift_msg.strip()


def generate_swift_logs(
    csv_path: str, output_txt_path: str, sample_size: int = 100
):
    """
    Samples transactions from SAML-D dataset and generates a combined SWIFT log file.
    Ensures a balanced representation of normal and suspicious (laundering) cases.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Source dataset not found at: {csv_path}")

    print(f"Reading raw transactions from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Separate normal and laundering cases
    normal_df = df[df["Is_laundering"] == 0]
    laundering_df = df[df["Is_laundering"] == 1]

    # Sample balanced dataset (80% normal, 20% laundering)
    n_laundering = min(len(laundering_df), int(sample_size * 0.2))
    n_normal = sample_size - n_laundering

    sampled_normal = normal_df.sample(n=n_normal, random_state=42)
    sampled_laundering = laundering_df.sample(n=n_laundering, random_state=42)

    combined_df = pd.concat([sampled_normal, sampled_laundering]).sample(
        frac=1, random_state=42
    )

    output_file = Path(output_txt_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating SWIFT text blocks for {len(combined_df)} transactions...")
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(combined_df.iterrows(), start=1):
            swift_block = format_swift_block(row.to_dict())
            f.write(f"--- TRANSACTION RECORD #{idx} ---\n")
            f.write(swift_block + "\n\n")

    print(
        f"✅ Successfully generated SWIFT transaction log at: {output_file.resolve()}"
    )


if __name__ == "__main__":
    raw_csv = "data/raw/saml_d_transactions.csv"
    processed_log = "data/processed/swift_transactions.txt"
    generate_swift_logs(
        csv_path=raw_csv, output_txt_path=processed_log, sample_size=100
    )