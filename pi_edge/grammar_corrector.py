"""
pi_edge/grammar_corrector.py
=============================
Grammar, hesitation, and accent corrector for human conversational speech:
1. Removes conversational hesitation tokens ('um', 'uh', 'erm', 'äh', 'euh')
2. Deduplicates stuttered words ('the the', 'I I', 'ich ich', 'je je')
3. Post-corrects common phonetic mishearings in travel/toll domains
4. Ensures proper sentence capitalization and punctuation
5. Enforces 3-language constraints (English, German, French)
"""

import re
from typing import Dict, Any, List, Optional

# Hesitation filler words to strip
FILLER_WORDS = {
    "uh", "um", "erm", "ah", "er", "mm", "hmm",
    "äh", "ähm", "öh", "halt",
    "euh", "bah", "ben"
}

# Domain vocabulary phonetic fixes (accent / mic distortion corrections)
PHONETIC_DOMAIN_FIXES = [
    # English fixes
    (r"\bhow much is the actual\b", "how much is the toll"),
    (r"\bthe actual\b", "the toll"),
    (r"\bthe actuals\b", "the tolls"),
    (r"\bwhere is the actual\b", "where is the toll"),
    (r"\bhighway actual\b", "highway toll"),
    (r"\bauto bahn\b", "autobahn"),

    # German fixes
    (r"\bhello we get\b", "Hallo, wie geht's"),
    (r"\bwe get\b", "wie geht's"),
    (r"\bwie gehts\b", "wie geht's"),
    (r"\bausweisterbein\b", "Ausweis dabei"),
    (r"\bdank\b", "danke"),
    (r"\bmaut station\b", "Mautstation"),
    (r"\bfahr karten\b", "Fahrkarten"),

    # French fixes
    (r"\bpeage\b", "péage"),
    (r"\bcombien coute le peage\b", "combien coûte le péage")
]


# Prompts or hallucination strings to discard completely
HALLUCINATION_PATTERNS = [
    r"tollmatcher",
    r"speech translation for human conversation",
    r"in english,? german,? and greetings",
    r"in english,? german,? and french",
    r"go to the book on this",
    r"based on them",
    r"thank you for watching",
    r"subtitles by",
    r"subscribe to",
    r"toll,?\s*highway,?\s*vignette",
    r"cash,?\s*contactless,?\s*payment",
    r"paris,?\s*credit card"
]


def clean_human_transcript(text: str, detected_lang: str) -> str:
    """
    Cleans raw human speech transcript from accents, stutters, and hesitation tokens.
    """
    if not text or not text.strip():
        return ""

    cleaned = text.strip()

    # 0a. Reject non-Latin characters (CJK, Arabic, Cyrillic) - strictly EN/DE/FR supported
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]', cleaned):
        return ""

    # 0b. Reject character-level repetition loops (e.g. 'aaaaa', 'weieiei...')
    if re.search(r'(.)\1{4,}', cleaned) or re.search(r'(.{2,6})\1{3,}', cleaned):
        return ""

    # 0c. Reject known Whisper prompt hallucinations or YouTube subtitle tags
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            return ""

    # 1. Remove repeated stuttered words ("the the", "I I", "ich ich")
    cleaned = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)

    # 2. Remove filler / hesitation tokens
    words = cleaned.split()
    filtered_words = [w for w in words if w.lower().strip(".,?!") not in FILLER_WORDS]
    if not filtered_words:
        return ""
    cleaned = " ".join(filtered_words)

    # 3. Apply domain phonetic corrections
    for pattern, replacement in PHONETIC_DOMAIN_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 4. Capitalize first letter and ensure clean sentence formatting
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # 5. Fix standalone English 'i' to 'I'
    cleaned = re.sub(r"\b(i)\b", "I", cleaned)

    # 6. Reject if length of actual letters is too short (< 2 alphanumeric characters)
    if len(re.findall(r"\w", cleaned)) < 2:
        return ""

    return cleaned


def resolve_three_way_language(
    detected_lang: str,
    all_language_probs: Optional[List[tuple]] = None,
    default_lang: str = "en"
) -> str:
    """
    Restricts language detection strictly to the 3 supported languages:
    English ('en'), German ('de'), French ('fr').
    Prevents Whisper from accidentally classifying background noise or accents
    into unsupported languages (like Korean, Chinese, Arabic, etc.).
    """
    supported = {"en", "de", "fr"}
    if detected_lang in supported:
        return detected_lang

    # If all language probabilities are provided, pick the highest among {en, de, fr}
    if all_language_probs:
        prob_dict = dict(all_language_probs)
        best_candidate = max(supported, key=lambda l: prob_dict.get(l, 0.0))
        if prob_dict.get(best_candidate, 0.0) > 0.05:
            return best_candidate

    return default_lang
