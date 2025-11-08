#!/usr/bin/env python3
"""
Download checkpoints and datasets from GitHub Releases or Zenodo.

Usage:
    python scripts/download_artifacts.py --checkpoint run_20251020_104533 --source github
    python scripts/download_artifacts.py --checkpoint run_20251020_104533 --source zenodo --doi 10.5281/zenodo.XXXXX
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(url: str, output_path: Path, expected_hash: str | None = None) -> bool:
    """
    Download file from URL with progress bar and optional hash verification.

    Args:
        url: URL to download from
        output_path: Where to save the file
        expected_hash: Expected SHA256 hash (optional)

    Returns:
        True if download successful
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            open(output_path, "wb") as f,
            tqdm(
                desc=output_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        # Verify hash if provided
        if expected_hash:
            print("Verifying checksum...")
            actual_hash = compute_sha256(output_path)
            if actual_hash != expected_hash:
                print("✗ Hash mismatch!")
                print(f"  Expected: {expected_hash}")
                print(f"  Got:      {actual_hash}")
                output_path.unlink()
                return False
            print("✓ Checksum verified")

        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_from_github(checkpoint_name: str, tag: str = "latest") -> bool:
    """Download checkpoint from GitHub Releases."""
    repo = "Marcux777/GNN_to_Knapsack"
    filename = f"{checkpoint_name}.tar.gz"

    if tag == "latest":
        url = f"https://github.com/{repo}/releases/latest/download/{filename}"
    else:
        url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"

    output_path = Path("checkpoints") / checkpoint_name
    archive_path = Path("checkpoints") / filename

    print(f"Downloading from GitHub: {url}")

    if download_file(url, archive_path):
        # Extract archive
        import tarfile

        print(f"Extracting to {output_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall("checkpoints")

        archive_path.unlink()
        print(f"✓ Downloaded and extracted to {output_path}")
        return True

    return False


def download_from_zenodo(doi: str, filename: str | None = None) -> bool:
    """Download from Zenodo using DOI."""
    # Extract record ID from DOI (e.g., 10.5281/zenodo.1234567 -> 1234567)
    record_id = doi.split(".")[-1]

    # Get record metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print("Fetching metadata from Zenodo...")

    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        metadata = response.json()

        files = metadata.get("files", [])
        if not files:
            print("✗ No files found in Zenodo record")
            return False

        # If filename specified, find it; otherwise download first file
        if filename:
            file_info = next((f for f in files if f["key"] == filename), None)
            if not file_info:
                print(f"✗ File '{filename}' not found in record")
                print(f"Available files: {[f['key'] for f in files]}")
                return False
        else:
            file_info = files[0]

        download_url = file_info["links"]["self"]
        checksum = file_info["checksum"].split(":")[-1]  # Extract hash from "md5:..."
        output_path = Path("checkpoints") / file_info["key"]

        print(f"Downloading: {file_info['key']}")
        return download_file(download_url, output_path, expected_hash=checksum)

    except Exception as e:
        print(f"✗ Zenodo download failed: {e}")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download checkpoints and datasets")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint name (e.g., run_20251020_104533)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["github", "zenodo"],
        default="github",
        help="Download source",
    )
    parser.add_argument("--tag", type=str, default="latest", help="GitHub release tag")
    parser.add_argument("--doi", type=str, help="Zenodo DOI (e.g., 10.5281/zenodo.1234567)")
    parser.add_argument("--filename", type=str, help="Specific filename to download from Zenodo")

    args = parser.parse_args()

    if args.source == "github":
        if not args.checkpoint:
            print("✗ --checkpoint required for GitHub downloads")
            sys.exit(1)
        success = download_from_github(args.checkpoint, args.tag)

    elif args.source == "zenodo":
        if not args.doi:
            print("✗ --doi required for Zenodo downloads")
            sys.exit(1)
        success = download_from_zenodo(args.doi, args.filename)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
