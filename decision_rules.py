"""Conservative decision safeguards for business entity matching."""

from train_model_v3 import normalize_text


def canonical_name_tokens(value):
    tokens = normalize_text(value).split()
    aliases = {
        "pvt": "private",
        "ltd": "limited",
        "inc": "incorporated",
        "corp": "corporation",
        "co": "company",
    }
    return [aliases.get(token, token) for token in tokens]


def canonical_country(value):
    country = normalize_text(value)
    aliases = {
        "us": "united states",
        "usa": "united states",
        "united states of america": "united states",
        "uk": "united kingdom",
        "great britain": "united kingdom",
    }
    return aliases.get(country, country)


def decide_match(name1, name2, country1, country2,
                 name_similarity, name_token_similarity,
                 model_score, threshold):
    """Return (decision, reason) using the app's conservative rules."""

    generic_name_words = {
        "of", "the", "and", "for",
        "college", "academy", "university",
        "engineering", "institute", "institution",
        "school", "technology", "technologies", "technological",
    }
    legal_name_words = {
        "private", "limited", "incorporated",
        "corporation", "company", "llc", "llp",
    }

    reference_words = (
        set(canonical_name_tokens(name1))
        - generic_name_words - legal_name_words
    )
    candidate_words = (
        set(canonical_name_tokens(name2))
        - generic_name_words - legal_name_words
    )

    country1_normalized = canonical_country(country1)
    country2_normalized = canonical_country(country2)

    country_conflict = (
        bool(country1_normalized)
        and bool(country2_normalized)
        and country1_normalized != country2_normalized
    )

    names_conflict = (
        bool(reference_words)
        and bool(candidate_words)
        and not (reference_words & candidate_words)
    ) or (
        name_similarity < 0.50
        and name_token_similarity < 0.50
    )

    if country_conflict:
        return (
            "REVIEW",
            "The country fields conflict. This may indicate different entities, "
            "a branch, or inconsistent source data; manual verification is required.",
        )

    if names_conflict:
        return (
            "REVIEW",
            "Business names have insufficient core-name agreement. "
            "A shared address alone is not enough to confirm identity.",
        )

    if model_score >= threshold:
        return (
            "MATCH",
            "The model score exceeds its diagnostic threshold and the safeguard "
            "found no blocking name or country conflict. Verify identity independently.",
        )

    return (
        "NON-MATCH",
        "The model score is below its diagnostic threshold. "
        "This result is not independently verified.",
    )
