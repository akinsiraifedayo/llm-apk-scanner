"""Classify the build toolchain of every LLM-integrated application.

Reads the APK's ZIP directory only, which is cheap and does not require
decoding resources. Classification is by marker file, in priority order, so an
application carrying both a Flutter engine and a React Native bundle is counted
once under the framework that owns its application logic.
"""
import json, sqlite3, zipfile, glob, sys
from collections import Counter
from pathlib import Path

DRIVE = "/media/crypt/94E8F52FE8F5106A/UEL-THESIS-APK-CORPUS-OFFLOADED-20260825"
SC = "validation"

# (label, predicate on the set of archive entry names). Order is priority.
def has(names, *frags):
    return any(any(f in n for f in frags) for n in names)

db = sqlite3.connect("/home/crypt/Desktop/UEL/dissertation/corpus/ledger_v2.db")
llm = {r[0].lower(): r[1] for r in db.execute(
    "select sha256, pkg_name from apk where state='scanned' and llm_integrated=1")}
paths = {Path(p).stem.lower(): p for p in
         glob.glob(f"{DRIVE}/staging/*.apk") + glob.glob(f"{DRIVE}/apks/*/*.apk")}

counts, rows, failed = Counter(), [], 0
for sha, pkg in llm.items():
    p = paths.get(sha)
    if not p:
        failed += 1; continue
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
    except Exception:
        failed += 1; continue
    if has(names, "libflutter.so", "flutter_assets/"):
        fw = "Flutter"
    elif has(names, "index.android.bundle", "libreactnativejni.so", "libhermes.so"):
        fw = "React Native"
    elif has(names, "libunity.so", "assets/bin/Data"):
        fw = "Unity"
    elif has(names, "libcordova", "www/cordova.js", "assets/www/index.html", "assets/capacitor.config.json"):
        fw = "Cordova / Capacitor"
    elif has(names, "libmonodroid.so", "assemblies/"):
        fw = "Xamarin / MAUI"
    elif has(names, "libkoin.so", "kotlin/kotlin.kotlin_builtins"):
        fw = "Native (Kotlin/Java)"
    else:
        fw = "Native (Kotlin/Java)"
    counts[fw] += 1
    rows.append({"sha256": sha, "package": pkg, "framework": fw})

n = sum(counts.values())
print(f"classified {n} of {len(llm)} LLM-integrated applications ({failed} unavailable)\n")
for fw, c in counts.most_common():
    print(f"  {fw:24} {c:4}  {c/n*100:5.1f}%")
cross = sum(c for f, c in counts.items() if f != "Native (Kotlin/Java)")
print(f"\n  cross-platform total   {cross:4}  {cross/n*100:5.1f}%")
json.dump({"n": n, "unavailable": failed, "counts": dict(counts), "rows": rows},
          open("validation/framework_census.json", "w"), indent=1)
