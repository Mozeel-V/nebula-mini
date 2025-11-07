#!/usr/bin/env python3
"""
Build API candidate set from benign traces (supports JSON events or
pipe/text sandbox traces like the provided Malware-Traces gw/mw files).
Writes JSON list of candidate "event" text snippets that can be sampled by attacks.

Usage:
    python api_candidates.py --raw_dir /path/to/Malware-Traces/gw --out checkpoints/api_candidates.json --top_k 200
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import re

FUNC_RE = re.compile(r'function\|(.*?)($|\|)')   # matches function|GetLastError
API_TOKEN_RE = re.compile(r'api:([A-Za-z0-9_]+)') # matches api:ReadFile
KW_FUNCNAME = ['function', 'api', 'call', 'target_func']

def extract_apis_from_json_file(p):
    try:
        j = json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        return []
    out = []
    # original format: each event is a dict with "api" field (api name)
    for ev in j.get("events", []):
        if isinstance(ev, dict):
            api = ev.get("api") or ev.get("function") or ev.get("name")
            if api:
                out.append(str(api))
        else:
            # fallback: if events are strings, try to find api: tokens
            if isinstance(ev, str):
                m = API_TOKEN_RE.search(ev)
                if m:
                    out.append(m.group(1))
    return out

def extract_apis_from_txt_file(p):
    apis = []
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                # case 1: pipe-separated key|value tokens -> look for function|NAME
                m = FUNC_RE.search(ln)
                if m:
                    apiname = m.group(1).strip()
                    if apiname:
                        apis.append(apiname)
                        continue
                # case 2: look for "api:Name" tokens in-line
                m2 = API_TOKEN_RE.findall(ln)
                for a in m2:
                    apis.append(a)
                # case 3: some logs may present "origin|...|target|...|function|Name" already covered,
                # but check for other patterns like "Call: Name(" or "-> Name("
                # simple heuristic: words with CamelCase or snake_case likely API names
                # split by non-alnum and pick likely API-like tokens (limited heuristics)
                parts = re.split(r'[^A-Za-z0-9_]', ln)
                for ptoken in parts:
                    if len(ptoken) >= 3 and (ptoken[0].isupper() or '_' in ptoken):
                        # filter common noise tokens
                        if ptoken.lower() in ('timestamp', 'pid', 'tid', 'origin', 'target', 'dll', 'when', 'result'):
                            continue
                        apis.append(ptoken)
    except Exception:
        pass
    return apis

def extract_apis_from_raw(raw_dir, top_k=100, only_benign=False, benign_dir_names=None):
    """
    raw_dir: path to directory containing logs (will be walked recursively)
    only_benign: if True and benign_dir_names provided, only collect from dirs matching those names
    benign_dir_names: list of substrings to identify benign folders (e.g., ['gw', 'good', 'benign'])
    """
    raw_dir = Path(raw_dir)
    cnt = Counter()
    for p in raw_dir.rglob("*"):
        if p.is_dir():
            continue
        # optional filter: if only_benign True and benign_dir_names set, skip others
        if only_benign and benign_dir_names:
            # check if any benign keyword is in the file path
            if not any(bn.lower() in str(p).lower() for bn in benign_dir_names):
                continue

        if p.suffix.lower() == ".json":
            apis = extract_apis_from_json_file(p)
        elif p.suffix.lower() in (".txt", ".log", ".trace"):
            apis = extract_apis_from_txt_file(p)
        else:
            # try to parse as text for unknown extensions
            apis = extract_apis_from_txt_file(p)

        for a in apis:
            # normalize: strip, remove surrounding parens, keep only function name (no path)
            a_norm = a.strip()
            # if something like C:\Windows\..., try to extract basename
            if "\\" in a_norm or "/" in a_norm:
                a_norm = Path(a_norm).name
            # remove trailing parentheses / args if present
            a_norm = re.sub(r'\(.*\)$', '', a_norm)
            if a_norm:
                cnt[a_norm] += 1

    most = [api for api, _ in cnt.most_common(top_k)]
    return most, cnt

# fallback common APIs (kept for backward compatibility)
FALLBACK = [
    "ReadFile", "CreateFileW", "WriteFile", "RegOpenKeyExW", "RegSetValueExW", "CloseHandle",
    "CreateProcess", "GetProcAddress", "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
    "connect", "send", "recv", "Sleep", "ListDirectory", "Stat", "QueryPerformanceCounter"
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw",
                   help="root directory of raw traces (walked recursively). Accepts JSON or txt logs.")
    p.add_argument("--out", default="checkpoints/api_candidates.json")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--only_benign", action="store_true",
                   help="if set, filter files by benign_dir_names (useful if you pass parent folder containing gw and mw)")
    p.add_argument("--benign_dir_names", default="gw,good,benign",
                   help="comma-separated substrings used to detect benign folders when --only_benign is set")
    p.add_argument("--min_count", type=int, default=1, help="min frequency for an API to be considered")
    args = p.parse_args()

    benign_dirs = [s.strip() for s in args.benign_dir_names.split(",") if s.strip()]

    cand, counts = extract_apis_from_raw(args.raw_dir, top_k=max(500, args.top_k*3),
                                        only_benign=args.only_benign,
                                        benign_dir_names=benign_dirs)
    # apply min_count filter and then top_k
    filtered = [c for c in cand if counts[c] >= args.min_count]
    if not filtered:
        filtered = FALLBACK[:args.top_k]
    events = [f"api:{api}" for api in filtered[: args.top_k]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    print(f"Wrote {len(events)} api candidates to {args.out}")

if __name__ == "__main__":
    main()
