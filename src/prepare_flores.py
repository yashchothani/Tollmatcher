#!/usr/bin/env python3
"""
src/prepare_flores.py
======================
Extracts and structures a balanced 36-sentence group pilot dataset from official
Meta FLORES-200 dev and devtest splits for the TollMatcher multilingual speech translation benchmark.

Target Languages:
- English (eng_Latn)
- German (deu_Latn)
- French (fra_Latn)

All 6 translation directions supported:
  EN->DE, DE->EN, EN->FR, FR->EN, DE->FR, FR->DE

Stratification:
- 12 Short (<12 words), 12 Medium (12-22 words), 12 Long (>22 words)
- 18 from 'dev', 18 from 'devtest'
- Topics: Travel/Wikivoyage, Health/Medicine, Disaster/Accident, Technology, Politics/Law, Sports/Society
- Difficulty: Easy (9), Medium (10), Hard (9), Stress Test (8)
"""

import json
import os
import re
import sys
from typing import Dict, List, Any


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FLORES_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw", "flores200")
OUTPUT_JSONL_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_dataset.jsonl")
OUTPUT_MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_manifest.json")
SCHEMA_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "schemas", "dataset.schema.json")


# Curated selections: list of tuples: (split, 0_indexed_line_num, length_cat, difficulty_cat, challenge_note)
CURATED_SELECTIONS = [
    # --- DEV SPLIT (18 groups: 6 Short, 6 Medium, 6 Long) ---
    # Short (<12 words)
    ("dev", 3, "short", "stress_test", "Military title and Thai proper name: Squadron Leader Dilokrit Pattavee"),
    ("dev", 4, "short", "stress_test", "Emergency vehicular lexicon: airport fire vehicle rolled over"),
    ("dev", 12, "short", "medium", "Sports proper noun and tournament entity: Raonic in Brisbane Open"),
    ("dev", 95, "short", "easy", "Formal declarative institutional investigation inquiry"),
    ("dev", 114, "short", "medium", "Corporate acquisition superlative entity: eBay history"),
    ("dev", 158, "short", "stress_test", "Numeral entity and ranking punctuation: Johnson second with 2,243 points"),

    # Medium (12-22 words)
    ("dev", 181, "medium", "medium", "Governmental policy terminology: Irish government parliamentary urgency"),
    ("dev", 655, "medium", "easy", "Travel advisory safety aphorism: cold weather only real danger"),
    ("dev", 657, "medium", "medium", "Medical sleep deprivation conditional: long summer days health issues"),
    ("dev", 664, "medium", "easy", "Cultural tourism travel register: Walt Disney World pilgrimage"),
    ("dev", 668, "medium", "medium", "Ticketing conditions and legal prohibition: non-transferable passes"),
    ("dev", 700, "medium", "medium", "Immigration customs phrasing: organized migratory flow between countries"),

    # Long (>22 words)
    ("dev", 0, "long", "hard", "Biomedical engineering breakthrough: Stanford cell diagnostic inkjet chip"),
    ("dev", 2, "long", "stress_test", "Aviation emergency report with timestamps: JAS 39C Gripen crash UTC"),
    ("dev", 649, "long", "stress_test", "Telecommunications infrastructure with statistics: Greenland 93% internet optic cable"),
    ("dev", 650, "long", "hard", "Travel accommodation connectivity: guesthouse wifi and internet cafes"),
    ("dev", 656, "long", "hard", "Arctic climate travel advisory: cold seasons clothing recommendation"),
    ("dev", 672, "long", "hard", "Park reservation logistics: Bright Angel Campground Phantom Ranch bookings"),

    # --- DEVTEST SPLIT (18 groups: 6 Short, 6 Medium, 6 Long) ---
    # Short (<12 words)
    ("devtest", 6, "short", "easy", "Smart home device IoT phrasing: WiFi door bell"),
    ("devtest", 8, "short", "stress_test", "Broadcast entity and date: late 2017 Siminoff QVC channel"),
    ("devtest", 34, "short", "stress_test", "Public safety numerical casualties: nineteen injured police officers"),
    ("devtest", 39, "short", "easy", "Travel misadventure narrative: Iwasaki trouble during trip"),
    ("devtest", 93, "short", "easy", "Industrial incident visual observation: white smoke from plant"),
    ("devtest", 282, "short", "easy", "Emergency medical trauma statement: driver serious head injuries"),

    # Medium (12-22 words)
    ("devtest", 0, "medium", "medium", "Scientific laboratory speech quote: 4-month-old non-diabetic mice"),
    ("devtest", 77, "medium", "stress_test", "Disaster precaution numeral entity: fourteen Hawaii coastline schools closed"),
    ("devtest", 692, "medium", "medium", "Bilingual national capital tourism: Ottawa museums showcase Canada"),
    ("devtest", 695, "medium", "easy", "Geographic wilderness description: northern sparsely populated territory"),
    ("devtest", 697, "medium", "easy", "Maritime geographical location: East African Islands Indian Ocean"),
    ("devtest", 699, "medium", "medium", "Political geography resort status: independent nations and French associates"),

    # Long (>22 words)
    ("devtest", 1, "long", "hard", "Clinical medicine academic disclaimer: Dr Ehud Ur Dalhousie University"),
    ("devtest", 75, "long", "hard", "Oceanographic tsunami bulletin: sea level readings Pago Pago and Niue"),
    ("devtest", 76, "long", "hard", "Telecommunications disaster blackout: Tonga power loss PTWC tsunami warning"),
    ("devtest", 707, "long", "stress_test", "Archaeological historical navigation: Casablanca Berber fishermen 10th century BCE"),
    ("devtest", 714, "long", "medium", "Regional security travel warning: Goma North Kivu safety assessment"),
    ("devtest", 726, "long", "hard", "National park tourism pass logistics: South Africa Wild Card entry")
]


def load_split_data(split: str) -> Dict[str, Any]:
    """Loads text files and metadata TSV for a split."""
    en_file = os.path.join(RAW_FLORES_DIR, split, f"eng_Latn.{split}")
    de_file = os.path.join(RAW_FLORES_DIR, split, f"deu_Latn.{split}")
    fr_file = os.path.join(RAW_FLORES_DIR, split, f"fra_Latn.{split}")
    meta_file = os.path.join(RAW_FLORES_DIR, f"metadata_{split}.tsv")

    with open(en_file, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]
    with open(de_file, "r", encoding="utf-8") as f:
        de_lines = [line.strip() for line in f]
    with open(fr_file, "r", encoding="utf-8") as f:
        fr_lines = [line.strip() for line in f]

    meta_rows = []
    with open(meta_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            row = dict(zip(header, parts))
            meta_rows.append(row)

    assert len(en_lines) == len(de_lines) == len(fr_lines) == len(meta_rows), \
        f"Mismatch in line counts for split {split}!"

    return {
        "en": en_lines,
        "de": de_lines,
        "fr": fr_lines,
        "metadata": meta_rows
    }


def validate_record_against_schema(record: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validates record against dataset.schema.json definition."""
    text_schema = schema["$defs"]["aligned_text_group"]
    required_fields = text_schema["required"]
    for field in required_fields:
        if field not in record:
            raise ValueError(f"Record {record.get('sentence_group_id')} missing required field: {field}")

    # Check consts & enums
    if record["record_type"] != "aligned_text_group":
        raise ValueError("record_type must be 'aligned_text_group'")
    if not re.match(text_schema["properties"]["sentence_group_id"]["pattern"], record["sentence_group_id"]):
        raise ValueError(f"Invalid sentence_group_id format: {record['sentence_group_id']}")
    if record["source_dataset"] not in text_schema["properties"]["source_dataset"]["enum"]:
        raise ValueError(f"Invalid source_dataset: {record['source_dataset']}")
    if record["original_split"] not in text_schema["properties"]["original_split"]["enum"]:
        raise ValueError(f"Invalid original_split: {record['original_split']}")
    if record["sentence_length_category"] not in text_schema["properties"]["sentence_length_category"]["enum"]:
        raise ValueError(f"Invalid sentence_length_category: {record['sentence_length_category']}")
    if record["linguistic_difficulty_category"] not in text_schema["properties"]["linguistic_difficulty_category"]["enum"]:
        raise ValueError(f"Invalid linguistic_difficulty_category: {record['linguistic_difficulty_category']}")
    if record["translation_verification_status"] not in text_schema["properties"]["translation_verification_status"]["enum"]:
        raise ValueError(f"Invalid translation_verification_status: {record['translation_verification_status']}")
    if record["license_or_provenance_reference"] not in text_schema["properties"]["license_or_provenance_reference"]["enum"]:
        raise ValueError(f"Invalid license: {record['license_or_provenance_reference']}")

    # Check language codes
    expected_codes = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn"}
    if record["language_codes"] != expected_codes:
        raise ValueError(f"Language codes mismatch: {record['language_codes']}")

    # Check supported directions (all 6 required)
    expected_dirs = ["EN->DE", "DE->EN", "EN->FR", "FR->EN", "DE->FR", "FR->DE"]
    if sorted(record["supported_directions"]) != sorted(expected_dirs):
        raise ValueError(f"Supported directions must include all 6 pairs: {record['supported_directions']}")

    # Ensure non-empty text
    for lang in ["text_en", "text_de", "text_fr"]:
        if not record[lang] or len(record[lang].strip()) == 0:
            raise ValueError(f"Empty text field: {lang}")


def prepare_pilot_dataset() -> List[Dict[str, Any]]:
    """Builds and validates the 36-sentence pilot dataset."""
    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)

    print("Loading official FLORES-200 raw data...")
    data_dev = load_split_data("dev")
    data_devtest = load_split_data("devtest")

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    records = []
    group_num = 1

    for split, idx, len_cat, diff_cat, note in CURATED_SELECTIONS:
        split_data = data_dev if split == "dev" else data_devtest
        meta = split_data["metadata"][idx]

        group_id = f"TM_G{group_num:02d}"
        source_id = f"{split}_{idx+1:04d}"

        record = {
            "record_type": "aligned_text_group",
            "sentence_group_id": group_id,
            "source_dataset": "FLORES-200",
            "original_source_id": source_id,
            "original_split": split,
            "text_en": split_data["en"][idx],
            "text_de": split_data["de"][idx],
            "text_fr": split_data["fr"][idx],
            "language_codes": {
                "en": "eng_Latn",
                "de": "deu_Latn",
                "fr": "fra_Latn"
            },
            "supported_directions": [
                "EN->DE", "DE->EN", "EN->FR", "FR->EN", "DE->FR", "FR->DE"
            ],
            "topic": meta.get("topic"),
            "domain": meta.get("domain"),
            "url": meta.get("URL"),
            "sentence_length_category": len_cat,
            "linguistic_difficulty_category": diff_cat,
            "translation_verification_status": "official_flores200_verified",
            "license_or_provenance_reference": "CC-BY-SA-4.0",
            "notes": note
        }

        # Strict schema validation
        validate_record_against_schema(record, schema)
        records.append(record)
        group_num += 1

    # Write out JSONL file
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate Manifest Summary
    manifest = {
        "dataset_name": "TollMatcher FLORES-200 Pilot Benchmark",
        "version": "1.0.0",
        "total_sentence_groups": len(records),
        "total_directional_evaluation_pairs": len(records) * 6,
        "languages": {
            "en": "eng_Latn",
            "de": "deu_Latn",
            "fr": "fra_Latn"
        },
        "supported_directions": ["EN->DE", "DE->EN", "EN->FR", "FR->EN", "DE->FR", "FR->DE"],
        "splits_distribution": {
            "dev": sum(1 for r in records if r["original_split"] == "dev"),
            "devtest": sum(1 for r in records if r["original_split"] == "devtest")
        },
        "sentence_length_distribution": {
            "short": sum(1 for r in records if r["sentence_length_category"] == "short"),
            "medium": sum(1 for r in records if r["sentence_length_category"] == "medium"),
            "long": sum(1 for r in records if r["sentence_length_category"] == "long")
        },
        "difficulty_distribution": {
            "easy": sum(1 for r in records if r["linguistic_difficulty_category"] == "easy"),
            "medium": sum(1 for r in records if r["linguistic_difficulty_category"] == "medium"),
            "hard": sum(1 for r in records if r["linguistic_difficulty_category"] == "hard"),
            "stress_test": sum(1 for r in records if r["linguistic_difficulty_category"] == "stress_test")
        },
        "domains_distribution": {
            domain: sum(1 for r in records if r["domain"] == domain)
            for domain in set(r["domain"] for r in records if r["domain"])
        },
        "licensing": {
            "license": "CC-BY-SA-4.0",
            "attribution": "FLORES-200 / NLLB Team et al. (2022), arXiv:2207.04672",
            "commercial_use_permitted": True
        }
    }

    with open(OUTPUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(records)} sentence groups:")
    print(f" - JSONL: {OUTPUT_JSONL_PATH}")
    print(f" - Manifest: {OUTPUT_MANIFEST_PATH}")
    return records


if __name__ == "__main__":
    records = prepare_pilot_dataset()
    print("Done!")
