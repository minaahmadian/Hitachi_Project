from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MODAL_PATTERNS = (
    ("MUST", re.compile(r"\bmust\b", re.IGNORECASE)),
    ("SHALL", re.compile(r"\bshall\b", re.IGNORECASE)),
    ("SHOULD", re.compile(r"\bshould\b", re.IGNORECASE)),
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_ID_FORMAT_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(slots=True)
class LogicalRule:
    rule_id: str
    clause_id: str
    clause_title: str
    modality: str
    requirement_text: str
    condition: str | None
    action: str
    category: str
    severity: str
    confidence: float
    source_document: str


def _load_clauses(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(normalized) if s.strip()]


def _clean_sentence(sentence: str) -> str:
    s = sentence.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("(cid:2)", " ")
    s = s.replace("", " ")
    s = re.sub(r"\.{5,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -;:,")


def _is_noisy_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return True

    # Common OCR/pagination artifacts.
    if "RIPRODUZIONE SU LICENZA" in s or "Copia concessa a" in s:
        return True
    if "(cid:" in s:
        return True
    if re.search(r"\bo n a i l a t i\b", s.lower()):
        return True
    # OCR artifact: long runs of single-letter tokens separated by spaces.
    if re.search(r"(?:\b[A-Za-z]\b\s+){8,}", s):
        return True

    # Too little alphabetic content.
    alpha_count = sum(1 for ch in s if ch.isalpha())
    if alpha_count < 18:
        return True

    # Dense symbol/bullet rows are likely tables or OCR noise.
    symbol_count = len(re.findall(r"[♦•\[\]\(\)\|]", s))
    if symbol_count >= 6:
        return True

    return False


def _split_condition_action(sentence: str, modality: str) -> tuple[str | None, str]:
    """
    Improved split:
    - if sentence starts with If/When/Where/As long as, condition is up to the first comma
      or up to the modal word, whichever gives a meaningful condition.
    - action keeps the normative part.
    """
    s = sentence.strip()
    lower = s.lower()
    modal_match = re.search(rf"\b{modality.lower()}\b", lower)
    if not modal_match:
        return None, s

    leading_conditional = re.match(r"^(if|when|where|as long as)\b", lower) is not None
    if not leading_conditional:
        return None, s

    comma_pos = s.find(",")
    modal_pos = modal_match.start()

    condition: str | None = None
    if comma_pos != -1 and comma_pos < modal_pos:
        condition = s[:comma_pos].strip()
        action = s[comma_pos + 1 :].strip()
    else:
        condition = s[:modal_pos].strip(" ,;:-")
        action = s[modal_pos:].strip()

    if not condition or len(condition) < 6:
        return None, s
    if not action:
        return None, s
    return condition, action


def _detect_modality(sentence: str) -> str | None:
    for label, pattern in MODAL_PATTERNS:
        if pattern.search(sentence):
            return label
    return None


def _extract_condition_action(sentence: str, modality: str) -> tuple[str | None, str]:
    """
    Heuristic split:
    - if sentence contains "if/when/where/as long as", keep pre-modal as condition
    - action is sentence with leading condition removed when possible
    """
    condition, action = _split_condition_action(sentence, modality)
    return condition, action


def _categorize_rule(clause_id: str, title: str, sentence: str) -> str:
    text = f"{clause_id} {title} {sentence}".lower()
    if any(k in text for k in ("test", "verification", "validation", "evidence", "report")):
        return "verification_and_testing"
    if any(k in text for k in ("traceability", "requirement", "configuration")):
        return "traceability_and_requirements"
    if any(k in text for k in ("maintenance", "integration", "architecture", "design")):
        return "lifecycle_engineering"
    if any(k in text for k in ("safety", "integrity", "hazard", "risk")):
        return "safety_assurance"
    return "general_compliance"


def _severity_from_modality(modality: str) -> str:
    if modality in {"MUST", "SHALL"}:
        return "HIGH"
    return "MEDIUM"


def _confidence_from_modality(modality: str) -> float:
    if modality == "MUST":
        return 0.97
    if modality == "SHALL":
        return 0.94
    return 0.83


def _iter_rules(clauses: Iterable[dict[str, object]]) -> list[LogicalRule]:
    rules: list[LogicalRule] = []
    rule_counter = 1

    for clause in clauses:
        clause_id = str(clause.get("clause_id", "")).strip()
        title = str(clause.get("title", "")).strip()
        text = str(clause.get("text", "")).strip()
        source_document = str(clause.get("source_document", "CEI EN 50128.pdf"))

        if not clause_id or not text or not CLAUSE_ID_FORMAT_RE.match(clause_id):
            continue

        for sentence in _split_sentences(text):
            sentence = _clean_sentence(sentence)
            if _is_noisy_sentence(sentence):
                continue

            modality = _detect_modality(sentence)
            if modality is None:
                continue

            # Keep meaningful normative statements only.
            if len(sentence) < 35:
                continue

            condition, action = _extract_condition_action(sentence, modality)
            action = _clean_sentence(action)
            if _is_noisy_sentence(action):
                continue

            category = _categorize_rule(clause_id, title, sentence)
            severity = _severity_from_modality(modality)
            confidence = _confidence_from_modality(modality)

            rules.append(
                LogicalRule(
                    rule_id=f"R-{rule_counter:05d}",
                    clause_id=clause_id,
                    clause_title=title,
                    modality=modality,
                    requirement_text=sentence,
                    condition=condition,
                    action=action,
                    category=category,
                    severity=severity,
                    confidence=confidence,
                    source_document=source_document,
                )
            )
            rule_counter += 1

    return rules


def extract_rules(clauses_path: Path, output_path: Path) -> list[LogicalRule]:
    clauses = _load_clauses(clauses_path)
    rules = _iter_rules(clauses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(rule) for rule in rules], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return rules


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    clauses_path = repo_root / "data" / "regulatory" / "cei_en_50128_clauses.json"
    output_path = repo_root / "data" / "regulatory" / "cei_en_50128_rules.json"

    rules = extract_rules(clauses_path=clauses_path, output_path=output_path)
    print(f"Extracted rules: {len(rules)}")
    print(f"Wrote: {output_path}")
    if rules:
        print(f"First rule: {rules[0].rule_id} ({rules[0].clause_id}, {rules[0].modality})")
        print(f"Last rule: {rules[-1].rule_id} ({rules[-1].clause_id}, {rules[-1].modality})")


if __name__ == "__main__":
    main()
