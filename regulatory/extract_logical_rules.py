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
    lower = sentence.lower()
    conditional_markers = (" if ", " when ", " where ", " as long as ")
    has_condition = any(marker in f" {lower} " for marker in conditional_markers)

    condition: str | None = None
    action = sentence

    # Split around first modal occurrence to isolate decision logic.
    modal_match = re.search(rf"\b{modality.lower()}\b", lower)
    if modal_match:
        pre = sentence[: modal_match.start()].strip(" ,;:-")
        post = sentence[modal_match.start() :].strip()
        if has_condition and pre:
            condition = pre
            action = post
        else:
            action = sentence

    return (condition if condition else None, action)


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

        if not clause_id or not text:
            continue

        for sentence in _split_sentences(text):
            modality = _detect_modality(sentence)
            if modality is None:
                continue

            condition, action = _extract_condition_action(sentence, modality)
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
