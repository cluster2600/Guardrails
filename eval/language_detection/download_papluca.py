import json
from pathlib import Path

TARGET_LANGUAGES = {"en", "es", "zh", "de", "fr", "hi", "ja", "ar", "th"}


def download_and_filter():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return None

    output_dir = Path(__file__).parent
    output_path = output_dir / "papluca_filtered.json"

    if output_path.exists():
        print(f"Dataset already downloaded at {output_path}")
        return output_path

    print("Downloading papluca/language-identification from Hugging Face...")
    dataset = load_dataset("papluca/language-identification")

    data = []

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        for item in split:
            lang = item["labels"]
            if lang in TARGET_LANGUAGES:
                data.append({
                    "text": item["text"],
                    "language": lang,
                    "split": split_name,
                    "source": "papluca",
                })

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

    return output_path


if __name__ == "__main__":
    download_and_filter()
