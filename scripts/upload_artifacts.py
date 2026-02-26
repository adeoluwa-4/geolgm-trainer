from __future__ import annotations

import argparse
from pathlib import Path

import boto3

from geolgm.tracking.db import insert_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--access-key", default=None)
    parser.add_argument("--secret-key", default=None)
    args = parser.parse_args()

    session = boto3.session.Session(
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
    )
    s3 = session.client("s3", endpoint_url=args.endpoint) if args.endpoint else session.client("s3")

    run_dir = Path("runs") / args.run_id / "artifacts"
    for path in run_dir.glob("*"):
        key = f"{args.run_id}/{path.name}"
        s3.upload_file(str(path), args.bucket, key)
        insert_artifact(Path("runs.db"), args.run_id, "s3", key)
        print(f"Uploaded {path} -> {key}")


if __name__ == "__main__":
    main()
