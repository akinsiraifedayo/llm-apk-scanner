"""Separate confidential system prompts from user-selectable presets.

METHODOLOGY.md §4.1.6 requires RQ1 statistics to report these two classes
separately, using three criteria: **register of address**, explicit
**confidentiality markers**, and **role binding**. The detector does not carry
that classification, so it is applied here, at analysis time, over the findings
the detector produced.

Why this reads the APK rather than the detail file: the ``evidence`` field is
truncated (60 characters in this corpus) and ends mid-sentence, so a register
judgement made from it would be made from a fragment. The rule established
after Defect 15 is that any rescoring uses the full string pulled from the
artefact, never the truncated evidence. This therefore re-extracts candidates
with the shipping detector and keeps the whole matched run.

Usage:
    python scripts/classify_prompts.py [--out validation/prompt_classes.json]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sqlite3
from pathlib import Path

from llm_apk_scanner.apk_loader import load_apk
from llm_apk_scanner.scanners.prompts import SystemPromptScanner

LEDGER = "corpus/ledger_v2.db"
DETAIL = "results/detail_v2"
STAGING = "corpus/staging"

# --- §4.1.6 criterion 1: register of address --------------------------------
# Second person, system addressing the model. The prompt speaks *to* the model
# about what the model is.
SECOND_PERSON = re.compile(
    r"\b(you are|you're|you will|you must|you should always|your task is|"
    r"your role is|your name is|act as if you)\b"
    r"|你是|您是|你的任务|你的角色|あなたは|당신은|أنت",
    re.IGNORECASE,
)
# First person, user addressing the model. The canonical form is the opener of
# the public CC0 prompt collection (awesome-chatgpt-prompts), which ships
# verbatim inside preset pickers.
FIRST_PERSON = re.compile(
    r"\b(i want you to act as|i want you to|i will (tell|give|type)|"
    r"my first (request|sentence|command) is)\b",
    re.IGNORECASE,
)

# --- §4.1.6 criterion 2: explicit confidentiality markers -------------------
CONFIDENTIAL = re.compile(
    r"never reveal|do not reveal|don'?t reveal|never disclose|do not disclose|"
    r"don'?t disclose|keep (these |this |your )?(instructions?|prompt|rules?)"
    r"[^.]{0,30}(secret|confidential|private)|"
    r"under no circumstances[^.]{0,60}(reveal|disclose|share)|"
    r"do not share (these|this|your) (instructions?|prompt|rules?)|"
    r"never (mention|tell|output|repeat)[^.]{0,40}(instructions?|prompt|system)|"
    r"not allowed to (reveal|discuss|share)|"
    r"ignore (any|all) (attempts?|requests?)[^.]{0,40}(reveal|instructions?)|"
    r"不要透露|不得透露|保密|禁止泄露",
    re.IGNORECASE,
)

# --- §4.1.6 criterion 3: role binding ---------------------------------------
# A named persona the model is bound to, as distinct from a generic role word.
# "You are OI, an AI assistant" binds; "you are a helpful assistant" does not.
ROLE_BINDING = re.compile(
    r"\byou are ([A-Z][\w'\-]{1,24})(,| the | an? (AI|assistant|chatbot|bot))"
    r"|\byour name is\b"
    r"|\bif (the user|anyone) asks who (made|created|built) you\b"
    r"|\bdeveloped by\b[^.]{0,40}\bdo not\b",
)

# White-labelling: the model instructed to misrepresent its own provenance.
# METHODOLOGY §4.1.6 calls this out as reportable in its own right.
WHITE_LABEL = re.compile(
    r"(asks|ask) who (made|created|built|developed) you[^.]{0,80}"
    r"(replace|say|answer|tell them|respond)"
    r"|never (say|mention|admit)[^.]{0,40}"
    r"(openai|chatgpt|gpt-|anthropic|claude|gemini|google)"
    r"|do not (say|mention|reveal)[^.]{0,30}(you are|you're) "
    r"(chatgpt|gpt|claude|gemini)",
    re.IGNORECASE,
)


def classify(text: str) -> dict:
    """Apply the three §4.1.6 criteria to one full prompt string."""
    first = bool(FIRST_PERSON.search(text))
    second = bool(SECOND_PERSON.search(text))
    confid = bool(CONFIDENTIAL.search(text))
    binding = bool(ROLE_BINDING.search(text))

    # Register decides, per §4.1.6, because it is the criterion that maps onto
    # provenance: first person is a user turn shipped as a preset, second
    # person is a developer instruction shipped as configuration. A string
    # carrying an explicit confidentiality marker is a system prompt whatever
    # its opener, because no preset library asserts confidentiality over
    # publicly published text.
    if confid:
        cls = "system_prompt"
    elif first and not second:
        cls = "preset"
    elif second and not first:
        cls = "system_prompt"
    elif first and second:
        # "I want you to act as X. You are ..." — the preset libraries do this.
        # The opener is what the string is; treat the leading register as
        # decisive and record the ambiguity for manual review.
        cls = "preset" if FIRST_PERSON.search(text).start() < SECOND_PERSON.search(text).start() else "system_prompt"
    else:
        cls = "unclassified"

    return {
        "class": cls,
        "register_first_person": first,
        "register_second_person": second,
        "confidentiality_marker": confid,
        "role_binding": binding,
        "white_labelling": bool(WHITE_LABEL.search(text)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="validation/prompt_classes.json")
    ap.add_argument(
        "--all-apps",
        action="store_true",
        help="classify every app with a system_prompt finding, not only the "
        "LLM-integrated ones (RQ1 uses only the latter)",
    )
    args = ap.parse_args()

    db = sqlite3.connect(LEDGER)
    llm = {
        r[0]
        for r in db.execute(
            "select sha256 from apk where state='scanned' and llm_integrated=1"
        )
    }
    pkgs = dict(db.execute("select sha256, pkg_name from apk"))

    # Which apps carry a system_prompt finding at all.
    targets = []
    for path in sorted(glob.glob(f"{DETAIL}/*.json.gz")):
        sha = Path(path).name[: -len(".json.gz")]
        if not args.all_apps and sha not in llm:
            continue
        detail = json.load(gzip.open(path, "rt"))
        n = sum(1 for f in detail["findings"] if f["kind"] == "system_prompt")
        if n:
            targets.append((sha, n))

    scanner = SystemPromptScanner()
    out = []
    for i, (sha, n_reported) in enumerate(targets, 1):
        apk = f"{STAGING}/{sha}.apk"
        if not Path(apk).exists():
            out.append({"sha256": sha, "pkg_name": pkgs.get(sha), "error": "apk not retained"})
            print(f"[{i}/{len(targets)}] {pkgs.get(sha)}: APK NOT RETAINED")
            continue

        corpus = load_apk(apk)
        strings = []
        for source_file, content in corpus:
            for candidate, _offset in scanner._extract_candidates(content):
                score = scanner._score(candidate, source_file=source_file)
                if score.score >= scanner.threshold:
                    strings.append({"source_file": source_file, "text": candidate})

        classed = [{**s, **classify(s["text"])} for s in strings]
        kinds = {c["class"] for c in classed}
        # An application is counted once per class, per reporting rule 4.
        app = {
            "sha256": sha,
            "pkg_name": pkgs.get(sha),
            "llm_integrated": sha in llm,
            "n_reported_in_run": n_reported,
            "n_rescored": len(classed),
            "has_system_prompt": "system_prompt" in kinds,
            "has_preset": "preset" in kinds,
            "has_unclassified": "unclassified" in kinds,
            "any_confidentiality_marker": any(c["confidentiality_marker"] for c in classed),
            "any_role_binding": any(c["role_binding"] for c in classed),
            "any_white_labelling": any(c["white_labelling"] for c in classed),
            "strings": classed,
        }
        out.append(app)
        flags = "".join(
            [
                "S" if app["has_system_prompt"] else "-",
                "P" if app["has_preset"] else "-",
                "?" if app["has_unclassified"] else "-",
                "C" if app["any_confidentiality_marker"] else "-",
                "W" if app["any_white_labelling"] else "-",
            ]
        )
        print(
            f"[{i}/{len(targets)}] {pkgs.get(sha):<45} {flags}  "
            f"run={n_reported:<3} rescored={len(classed)}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    real = [a for a in out if "error" not in a]
    print(f"\nwrote {args.out}")
    print(f"apps classified          : {len(real)}")
    print(f"  with confidential prompt: {sum(a['has_system_prompt'] for a in real)}")
    print(f"  with preset only        : "
          f"{sum(a['has_preset'] and not a['has_system_prompt'] for a in real)}")
    print(f"  with unclassified only  : "
          f"{sum(a['has_unclassified'] and not a['has_system_prompt'] and not a['has_preset'] for a in real)}")
    print(f"  asserting confidentiality: {sum(a['any_confidentiality_marker'] for a in real)}")
    print(f"  white-labelling          : {sum(a['any_white_labelling'] for a in real)}")


if __name__ == "__main__":
    main()
