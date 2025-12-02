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
    output_path = output_dir / "nemotron_filtered.json"

    if output_path.exists():
        print(f"Dataset already downloaded at {output_path}")
        return output_path

    print("Downloading nvidia/Nemotron-Safety-Guard-Dataset-v3 from Hugging Face...")
    dataset = load_dataset("nvidia/Nemotron-Safety-Guard-Dataset-v3")

    data = []
    redacted_by_lang = {}

    for split_name in dataset.keys():
        split = dataset[split_name]
        for item in split:
            lang = item["language"]
            prompt = item["prompt"]

            if lang not in TARGET_LANGUAGES:
                continue

            if prompt == "REDACTED" or not prompt:
                redacted_by_lang[lang] = redacted_by_lang.get(lang, 0) + 1
                continue

            data.append({
                "text": prompt,
                "language": lang,
                "split": split_name,
                "source": "nemotron",
                "prompt_label": item.get("prompt_label"),
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

    by_label = {}
    for item in data:
        label = item.get("prompt_label", "unknown")
        by_label[label] = by_label.get(label, 0) + 1

    print("\nSamples by safety label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")

    if redacted_by_lang:
        print("\nRedacted samples per language (excluded):")
        for lang, count in sorted(redacted_by_lang.items()):
            print(f"  {lang}: {count}")

    return output_path


if __name__ == "__main__":
    download_and_filter()
