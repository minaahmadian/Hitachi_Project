from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "shall",
    "must",
    "should",
    "into",
    "have",
    "been",
    "will",
    "are",
    "was",
    "were",
    "has",
    "had",
    "not",
    "its",
    "their",
    "than",
    "then",
    "also",
    "when",
    "where",
    "which",
    "while",
    "under",
    "over",
    "such",
    "each",
}


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    text: str
    source_type: str


@dataclass(slots=True)
class RuleEvaluation:
    rule_id: str
    clause_id: str
    modality: str
    severity: str
    status: str
    needs_derogation: bool
    matched_evidence_ids: list[str]
    rationale: str


@dataclass(slots=True)
class EvaluationSummary:
    total_rules_checked: int
    passed: int
    failed: int
    warning: int
    derogation_needed: int
    findings: list[RuleEvaluation]


class RegulatoryRuleEngine:
    def __init__(self, rules_path: Path) -> None:
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Rules file must be a JSON list")
        self.rules: list[dict[str, object]] = [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {w for w in words if w not in STOPWORDS}

    def _rule_keywords(self, rule: dict[str, object]) -> set[str]:
        clause = str(rule.get("clause_id", ""))
        title = str(rule.get("clause_title", ""))
        requirement = str(rule.get("requirement_text", ""))
        action = str(rule.get("action", ""))
        return self._tokenize(f"{clause} {title} {requirement} {action}")

    def _relevance_score(self, rule: dict[str, object], anomaly_text: str) -> float:
        rk = self._rule_keywords(rule)
        ak = self._tokenize(anomaly_text)
        if not rk or not ak:
            return 0.0
        overlap = len(rk.intersection(ak))
        return overlap / max(1, len(ak))

    def _match_evidence(self, rule: dict[str, object], evidence: list[EvidenceItem]) -> list[str]:
        rk = self._rule_keywords(rule)
        matched: list[str] = []
        for item in evidence:
            ek = self._tokenize(item.text)
            if not ek:
                continue
            overlap = len(rk.intersection(ek))
            ratio = overlap / max(1, len(rk))
            if overlap >= 2 and ratio >= 0.08:
                matched.append(item.evidence_id)
        return matched

    def evaluate(
        self,
        *,
        anomaly_text: str,
        evidence: list[EvidenceItem],
        top_k_rules: int = 30,
    ) -> EvaluationSummary:
        ranked = sorted(
            self.rules,
            key=lambda r: self._relevance_score(r, anomaly_text),
            reverse=True,
        )
        candidates = [r for r in ranked[:top_k_rules] if self._relevance_score(r, anomaly_text) > 0]

        findings: list[RuleEvaluation] = []
        for rule in candidates:
            matched = self._match_evidence(rule, evidence)
            modality = str(rule.get("modality", "SHOULD")).upper()
            severity = str(rule.get("severity", "MEDIUM")).upper()
            rule_id = str(rule.get("rule_id", ""))
            clause_id = str(rule.get("clause_id", ""))

            if matched:
                status = "PASS"
                needs_derogation = False
                rationale = f"Matched supporting evidence for {clause_id}: {', '.join(matched)}"
            else:
                if modality in {"MUST", "SHALL"}:
                    status = "FAIL"
                    needs_derogation = True
                    rationale = (
                        f"No supporting evidence found for mandatory rule {clause_id}; "
                        "derogation or corrective action is required."
                    )
                else:
                    status = "WARNING"
                    needs_derogation = False
                    rationale = f"No evidence found for recommended rule {clause_id}."

            findings.append(
                RuleEvaluation(
                    rule_id=rule_id,
                    clause_id=clause_id,
                    modality=modality,
                    severity=severity,
                    status=status,
                    needs_derogation=needs_derogation,
                    matched_evidence_ids=matched,
                    rationale=rationale,
                )
            )

        passed = sum(1 for f in findings if f.status == "PASS")
        failed = sum(1 for f in findings if f.status == "FAIL")
        warning = sum(1 for f in findings if f.status == "WARNING")
        derogation_needed = sum(1 for f in findings if f.needs_derogation)

        return EvaluationSummary(
            total_rules_checked=len(findings),
            passed=passed,
            failed=failed,
            warning=warning,
            derogation_needed=derogation_needed,
            findings=findings,
        )
