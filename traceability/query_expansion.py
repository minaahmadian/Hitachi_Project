"""
Deterministic domain query expansion for the APCS FIT corpus.

Why this exists
---------------
The RSSOM text mixes English prose with railway subsystem abbreviations
(PVIS, IAMS, MMIS, RTM, HMI, IL, ATS, FCS, OCC, …). User-phrased queries
often use the long form ("passenger information visualization system")
while the corpus uses the abbreviation ("PVIS"), or vice versa. A purely
semantic retriever will fail to match these pairs when embeddings are weak
(hash embedder) or when the training data has no railway vocabulary.

Query expansion adds the known short/long counterparts to the query so
both BM25 and vector paths see each version of the terminology. The
expansion is deterministic and auditable — the glossary below is the
single source of truth; every expansion is traceable.
"""
from __future__ import annotations

import re
from typing import Final


# ─────────────────────────────────────────────────────────────────────────────
# Domain glossary
# ─────────────────────────────────────────────────────────────────────────────
# Keys are the lowercase canonical form (short OR long); values are the other
# forms that should also appear in the query. Keep lowercase to match tokens.
# Source: RSSOM_APCS_FIT.docx and related project glossary.
# ─────────────────────────────────────────────────────────────────────────────

_BIDIR_GLOSSARY: Final[dict[str, list[str]]] = {
    # APCS = Asset Performance and Condition Supervision
    "apcs": ["asset performance condition supervision"],
    "asset performance condition supervision": ["apcs"],

    # MMIS = Maintenance Management Information System
    "mmis": ["maintenance management information system"],
    "maintenance management information system": ["mmis"],

    # IAMS = Integrated Asset Management System
    "iams": ["integrated asset management system"],
    "integrated asset management system": ["iams"],

    # PVIS = Passenger Visualization Information System (passenger info display)
    "pvis": ["passenger information visualization system", "passenger visualization information system"],
    "passenger information visualization system": ["pvis"],
    "passenger visualization information system": ["pvis"],

    # RTM = Real Time Monitoring
    "rtm": ["real time monitoring"],
    "real time monitoring": ["rtm"],

    # HMI = Human Machine Interface
    "hmi": ["human machine interface"],
    "human machine interface": ["hmi"],

    # IL = Integration Layer
    "il": ["integration layer"],
    "integration layer": ["il"],

    # ATS = Automatic Train Supervision
    "ats": ["automatic train supervision"],
    "automatic train supervision": ["ats"],

    # FCS = Facility Control System
    "fcs": ["facility control system"],
    "facility control system": ["fcs"],

    # OCC / iOCC / i-OCC = (integrated) Operations Control Centre
    "occ": ["operations control centre", "operations control center"],
    "iocc": ["i-occ", "integrated operations control centre"],
    "i-occ": ["iocc", "integrated operations control centre"],
    "integrated operations control centre": ["iocc", "i-occ"],

    # SCADA, TCMS, CBTC, PSD — extra domain terms
    "scada": ["supervisory control and data acquisition"],
    "tcms": ["train control and management system"],
    "cbtc": ["communication based train control"],
    "psd": ["platform screen door"],
    "platform screen door": ["psd"],

    # Verification vocabulary
    "fit": ["factory integration test"],
    "factory integration test": ["fit"],
    "fat": ["factory acceptance test"],
    "factory acceptance test": ["fat"],
    "sat": ["site acceptance test"],
    "site acceptance test": ["sat"],
}

# Crowding-level synonym cluster. The FIT matrix uses "empty / light / medium /
# high / full" for five distinct requirements; user queries often say
# "threshold" or "alert" or "crowding level" without the specific word.
_CROWDING_SYNONYMS: Final[list[str]] = [
    "crowding",
    "passenger density",
    "occupancy level",
    "occupancy threshold",
]

_VERIFICATION_SYNONYMS: Final[list[str]] = [
    "verification",
    "validation",
    "test evidence",
    "pass fail",
    "acceptance criteria",
]


_WORD_RX = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def expand_query(query: str, *, include_verification_vocab: bool = False) -> str:
    """
    Return the query augmented with glossary counterparts.

    The original query text is preserved and the expanded terms are appended
    so the expansion never removes or reorders caller-supplied tokens.

    Parameters
    ----------
    query : raw natural-language query
    include_verification_vocab : if True, also append verification synonyms
        (useful when the query is about a requirement's verdict, not just
        its description)
    """
    base = (query or "").strip()
    if not base:
        return base

    q_lower = base.lower()
    additions: list[str] = []
    seen: set[str] = set()

    # Direct abbreviation / long-form matches.
    for key, expansions in _BIDIR_GLOSSARY.items():
        if _contains_term(q_lower, key):
            for form in expansions:
                if form not in q_lower and form not in seen:
                    additions.append(form)
                    seen.add(form)

    # Crowding cluster: if any crowding-related term appears, add the others.
    if any(_contains_term(q_lower, syn) for syn in _CROWDING_SYNONYMS):
        for syn in _CROWDING_SYNONYMS:
            if syn not in q_lower and syn not in seen:
                additions.append(syn)
                seen.add(syn)

    if include_verification_vocab:
        for syn in _VERIFICATION_SYNONYMS:
            if syn not in q_lower and syn not in seen:
                additions.append(syn)
                seen.add(syn)

    if not additions:
        return base
    return f"{base} {' '.join(additions)}"


def _contains_term(haystack: str, term: str) -> bool:
    """Whole-word containment check. Avoids matching 'il' inside 'still'."""
    if " " in term or "-" in term:
        # Multi-word phrase: substring check is fine.
        return term in haystack
    pattern = r"\b" + re.escape(term) + r"\b"
    return re.search(pattern, haystack) is not None
