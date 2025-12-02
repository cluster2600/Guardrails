import io
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen

WILI_URL = "https://zenodo.org/record/841984/files/wili-2018.zip"

LANGUAGE_MAP = {
    "eng": "en",
    "spa": "es",
    "zho": "zh",
    "deu": "de",
    "fra": "fr",
    "hin": "hi",
    "jpn": "ja",
    "ara": "ar",
    "tha": "th",
}

TARGET_LANGUAGES = set(LANGUAGE_MAP.keys())


def download_and_extract():
    output_dir = Path(__file__).parent
    x_train_path = output_dir / "x_train.txt"

    if x_train_path.exists():
        print(f"WiLI-2018 already downloaded at {output_dir}")
        return output_dir

    print(f"Downloading WiLI-2018 from {WILI_URL}...")
    with urlopen(WILI_URL) as response:
        zip_data = response.read()

    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        zf.extractall(output_dir)

    print(f"Extracted to {output_dir}")
    return output_dir


def load_wili_data(wili_dir: Path):
    x_train_path = wili_dir / "x_train.txt"
    y_train_path = wili_dir / "y_train.txt"
    x_test_path = wili_dir / "x_test.txt"
    y_test_path = wili_dir / "y_test.txt"

    data = []

    for x_path, y_path, split in [
        (x_train_path, y_train_path, "train"),
        (x_test_path, y_test_path, "test"),
    ]:
        with open(x_path, "r", encoding="utf-8") as fx, open(y_path, "r", encoding="utf-8") as fy:
            for text, lang in zip(fx, fy):
                text = text.strip()
                lang = lang.strip()
                if lang in TARGET_LANGUAGES:
                    data.append({
                        "text": text,
                        "language": LANGUAGE_MAP[lang],
                        "split": split,
                        "source": "wili-2018",
                    })

    return data


def save_filtered_dataset(data, output_path: Path):
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} samples to {output_path}")

    by_lang = {}
    for item in data:
        lang = item["language"]
        by_lang[lang] = by_lang.get(lang, 0) + 1

    print("\nSamples per language:")
    for lang, count in sorted(by_lang.items()):
        print(f"  {lang}: {count}")


def main():
    wili_dir = download_and_extract()
    print("\nLoading and filtering for target languages...")
    data = load_wili_data(wili_dir)
    output_path = Path(__file__).parent / "wili_filtered.json"
    save_filtered_dataset(data, output_path)


if __name__ == "__main__":
    main()
