
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    ROOT
    / "data"
    / "azure_trace_raw"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
    / "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_PATH = ROOT / "data" / "azure_serverless_workload.csv"

CHUNK_SIZE = 500_000
BUCKET_SECONDS = 300  # Five minutes


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Azure trace not found:\n{RAW_PATH}"
        )

    print("Reading Azure Functions invocation trace...")
    print(f"Input: {RAW_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # First pass: find the earliest and latest timestamps.
    min_timestamp = np.inf
    max_timestamp = -np.inf
    total_rows = 0

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=["end_timestamp", "duration"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["end_timestamp"] = pd.to_numeric(
            chunk["end_timestamp"], errors="coerce"
        )
        chunk["duration"] = pd.to_numeric(
            chunk["duration"], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["end_timestamp", "duration"]
        )

        if chunk.empty:
            continue

        min_timestamp = min(
            min_timestamp, chunk["end_timestamp"].min()
        )
        max_timestamp = max(
            max_timestamp, chunk["end_timestamp"].max()
        )
        total_rows += len(chunk)

    if total_rows == 0:
        raise ValueError("No valid invocation records found.")

    print(f"Valid records: {total_rows:,}")
    print(f"Earliest completion: {min_timestamp:.3f} seconds")
    print(f"Latest completion: {max_timestamp:.3f} seconds")

    # Second pass: aggregate invocations and execution duration.
    # Each record represents one function invocation.
    bucket_counts = {}
    bucket_duration_sum = {}
    bucket_duration_count = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW_PATH,
            usecols=["end_timestamp", "duration"],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk["end_timestamp"] = pd.to_numeric(
            chunk["end_timestamp"], errors="coerce"
        )
        chunk["duration"] = pd.to_numeric(
            chunk["duration"], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["end_timestamp", "duration"]
        )

        if chunk.empty:
            continue

        # Convert elapsed completion time to five-minute buckets.
        bucket = np.floor(
            (chunk["end_timestamp"] - min_timestamp)
            / BUCKET_SECONDS
        ).astype(np.int64)

        grouped = pd.DataFrame({
            "bucket": bucket,
            "duration": chunk["duration"].to_numpy(),
        }).groupby("bucket").agg(
            invocation_count=("duration", "size"),
            duration_sum=("duration", "sum"),
            duration_count=("duration", "count"),
        )

        for bucket_id, row in grouped.iterrows():
            key = int(bucket_id)

            bucket_counts[key] = (
                bucket_counts.get(key, 0)
                + int(row["invocation_count"])
            )
            bucket_duration_sum[key] = (
                bucket_duration_sum.get(key, 0.0)
                + float(row["duration_sum"])
            )
            bucket_duration_count[key] = (
                bucket_duration_count.get(key, 0)
                + int(row["duration_count"])
            )

        if chunk_number % 5 == 0:
            print(f"Processed chunks: {chunk_number}")

    # Include empty buckets so zero-invocation intervals are represented.
    max_bucket = int(
        np.floor(
            (max_timestamp - min_timestamp) / BUCKET_SECONDS
        )
    )

    bucket_ids = np.arange(max_bucket + 1, dtype=np.int64)

    invocation_counts = np.array(
        [bucket_counts.get(int(i), 0) for i in bucket_ids],
        dtype=np.int64,
    )

    duration_sums = np.array(
        [bucket_duration_sum.get(int(i), 0.0) for i in bucket_ids],
        dtype=np.float64,
    )

    duration_counts = np.array(
        [bucket_duration_count.get(int(i), 0) for i in bucket_ids],
        dtype=np.int64,
    )

    mean_duration = np.divide(
        duration_sums,
        duration_counts,
        out=np.zeros_like(duration_sums),
        where=duration_counts > 0,
    )

    # Invocation rate is represented as invocations per five-minute bucket.
    result = pd.DataFrame({
        "time_idx": bucket_ids,
        "timestamp_seconds": (
            min_timestamp + bucket_ids * BUCKET_SECONDS
        ),
        "invocation_count": invocation_counts,
        "invocation_rate": invocation_counts.astype(float),
        "mean_duration_seconds": mean_duration,
    })

    # Calendar features are intentionally omitted because the raw
    # timestamps are elapsed seconds, not wall-clock timestamps.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Azure workload summary ===")
    print(f"Five-minute buckets: {len(result):,}")
    print(f"Total invocations: {invocation_counts.sum():,}")
    print(f"Mean invocations per bucket: {invocation_counts.mean():.2f}")
    print(f"Maximum invocations per bucket: {invocation_counts.max():,}")
    print(
        "Mean execution duration (seconds): "
        f"{mean_duration[duration_counts > 0].mean():.4f}"
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
