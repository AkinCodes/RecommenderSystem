"""Download MovieLens 100K dataset if not already present."""

import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k.zip")
REQUIRED_FILE = os.path.join(DATA_DIR, "u.data")


def download_and_extract():
    if os.path.exists(REQUIRED_FILE):
        print(f"  Dataset already exists at {REQUIRED_FILE}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"  Downloading MovieLens 100K from {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"  Downloaded to {ZIP_PATH}")

    print("  Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for member in zf.namelist():
            # Files inside the zip are under ml-100k/ — extract to data/
            if member.startswith("ml-100k/") and not member.endswith("/"):
                filename = os.path.basename(member)
                target = os.path.join(DATA_DIR, filename)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    os.remove(ZIP_PATH)
    print(f"  Extracted to {DATA_DIR}")

    if not os.path.exists(REQUIRED_FILE):
        raise FileNotFoundError(f"Expected {REQUIRED_FILE} after extraction but it's missing.")

    print("  Dataset ready.")


if __name__ == "__main__":
    download_and_extract()
