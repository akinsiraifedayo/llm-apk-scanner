"""Publish the analysis outputs without anything that identifies an application.

Reads the identified outputs the analysis scripts write under ``validation/``
and writes copies to ``findings/`` with application identifiers removed:

- ``chapter6_stats.json``: the per-credential ``package`` fields are dropped
  from the embedded liveness record. Everything else is already aggregate.
- ``liveness_results.json``: provider, fingerprint and disposition only, as in
  Appendix D. The fingerprint-to-package mapping stays private, and check
  times are cut to the date, since their order follows the SHA-256 order in
  which the applications were checked.
- ``prompt_classes.json``: classification flags per application and per
  string. SHA-256, package name and prompt text are dropped, and each string's
  source file is reduced to its container type, because names such as hashed
  web-bundle chunks are unique to one application.
- ``framework_census.json``: toolchain counts only; the per-application rows
  are dropped.

Records are ordered by their own content, never by SHA-256, so row order
cannot be matched against the published manifest. No per-application findings
table is published: combined with the manifest's year and store columns, the
smaller strata would narrow a finding to a handful of named applications.

Usage:
    python scripts/anonymise_findings.py [--source validation] [--out findings]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Per-application fields of ``prompt_classes.json`` that identify nothing.
APP_FIELDS = (
    "llm_integrated",
    "n_reported_in_run",
    "n_rescored",
    "has_system_prompt",
    "has_preset",
    "has_unclassified",
    "any_confidentiality_marker",
    "any_role_binding",
    "any_white_labelling",
    "error",
)

#: Per-string fields dropped from ``prompt_classes.json``.
STRING_DROPPED = ("source_file", "text")


def container(source_file: str) -> str:
    """Reduce an archive path to the kind of container it names."""
    if re.fullmatch(r"classes\d*\.dex", source_file):
        return "dex"
    if source_file.startswith("lib/") and source_file.endswith(".so"):
        return "native_library"
    if source_file == "resources.arsc":
        return "resource_table"
    if source_file == "AndroidManifest.xml":
        return "manifest"
    if source_file.startswith("assets/"):
        return "asset"
    if source_file.startswith("res/"):
        return "resource"
    return "other"


def by_content(record: dict) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def anonymise_credential(outcome: dict) -> dict:
    out = {k: v for k, v in outcome.items() if k != "package"}
    if out.get("checked_at"):
        out["checked_at"] = out["checked_at"][:10]
    return out


def anonymise_liveness(liveness: dict) -> dict:
    out = {}
    for key, value in liveness.items():
        if key in ("outcomes", "to_disclose"):
            value = sorted(
                (anonymise_credential(o) for o in value),
                key=lambda o: (o["provider"], o["state"], o["fingerprint"]),
            )
        out[key] = value
    return out


def anonymise_app(app: dict) -> dict:
    out = {k: app[k] for k in APP_FIELDS if k in app}
    strings = [
        {"container": container(s["source_file"]),
         **{k: v for k, v in s.items() if k not in STRING_DROPPED}}
        for s in app.get("strings", [])
    ]
    out["strings"] = sorted(strings, key=by_content)
    return out


def write(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=Path("validation"))
    ap.add_argument("--out", type=Path, default=Path("findings"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stats = json.loads((args.source / "chapter6_stats.json").read_text(encoding="utf-8"))
    if stats.get("liveness"):
        stats["liveness"] = anonymise_liveness(stats["liveness"])
    write(stats, args.out / "chapter6_stats.json")

    liveness = json.loads((args.source / "liveness_results.json").read_text(encoding="utf-8"))
    write(anonymise_liveness(liveness), args.out / "liveness_results.json")

    apps = json.loads((args.source / "prompt_classes.json").read_text(encoding="utf-8"))
    write(sorted((anonymise_app(a) for a in apps), key=by_content),
          args.out / "prompt_classes.json")

    census = json.loads((args.source / "framework_census.json").read_text(encoding="utf-8"))
    write({k: census[k] for k in ("n", "unavailable", "counts")},
          args.out / "framework_census.json")


if __name__ == "__main__":
    main()
