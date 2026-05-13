#!/usr/bin/env python3
"""Inspect derma_data.zip and repo layout so we can fix paths. Run in Colab after upload."""

from pathlib import Path

def main() -> None:
    repo = Path("/content/DermaFusion")
    for zip_name in ["derma_data.zip", "data.zip"]:
        for base in [Path("/content"), repo]:
            z = base / zip_name
            if z.exists():
                print(f"Found zip: {z}\n")
                import zipfile
                with zipfile.ZipFile(z, "r") as f:
                    names = f.namelist()
                print(f"Total entries: {len(names)}")
                print("First 40 paths in zip:")
                for n in names[:40]:
                    print(f"  {n}")
                if len(names) > 40:
                    print(f"  ... and {len(names) - 40} more")
                # Look for metadata and images
                csvs = [n for n in names if n.endswith(".csv")]
                train_like = [n for n in names if "train" in n.lower() and (".jpg" in n or "/" in n)]
                print(f"\nCSV files ({len(csvs)}): {csvs[:15]}")
                print(f"Paths containing 'train' (first 10): {train_like[:10]}")
                return

    print("No zip found in /content/ or repo. Listing /content:")
    for p in sorted(Path("/content").iterdir()):
        print(f"  {p}")
    if (repo / "data").exists():
        print(f"\n{repo}/data contents:")
        for p in sorted((repo / "data").rglob("*"))[:60]:
            print(f"  {p.relative_to(repo)}")

if __name__ == "__main__":
    main()
