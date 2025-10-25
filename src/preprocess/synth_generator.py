#!/usr/bin/env python3
import json, random
from pathlib import Path
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
def make_event(api=None, args=None, file=None, ip=None, domain=None, reg=None, hashv=None, message=None):
    ev = {}
    if api: ev["api"] = api
    if args: ev["args"] = args
    if file: ev["file"] = file
    if ip: ev["ip"] = ip
    if domain: ev["domain"] = domain
    if reg: ev["reg"] = reg
    if hashv: ev["hash"] = hashv
    if message: ev["message"] = message
    return ev
BENIGN_EVENTS = [
    lambda i: make_event("ReadFile", {"path": f"C:\\Windows\\Temp\\f{i}.tmp"}),
    lambda i: make_event("GetTickCount"),
    lambda i: make_event("RegOpenKeyExW", {"path": r"HKEY_LOCAL_MACHINE\\Software\\Microsoft"}),
    lambda i: make_event("CloseHandle"),
    lambda i: make_event("CreateFileW", {"path": r"C:\\Users\\Alice\\Documents\\doc.txt"}),
    lambda i: make_event("Sleep", {"ms": 100}),
    lambda i: make_event("connect", {"ip": "127.0.0.1"}),
]
MALICIOUS_SUBSEQ = [
    make_event("VirtualAllocEx", {"pid": 1000}),
    make_event("WriteProcessMemory", {"pid": 1000}),
    make_event("CreateRemoteThread", {"pid": 1000}),
    make_event("connect", {"ip": "95.211.198.12"}),
    make_event("send", {"domain": "malicious-c2.example.com"}),
]
def gen_long_trace(malicious=False, n_events=200, seed=None):
    if seed is not None: random.seed(seed)
    events = []
    for i in range(n_events):
        ev_fn = random.choice(BENIGN_EVENTS)
        events.append(ev_fn(i))
    if malicious:
        pos = random.randint(max(1, n_events//4), min(n_events-2, 3*n_events//4))
        for j, mev in enumerate(MALICIOUS_SUBSEQ):
            events.insert(pos + j, mev)
    return {"events": events}
def main(out_dir="data/raw", n_samples=200, mal_frac=0.5, n_events=200):
    out_dir = Path(out_dir)
    n_mal = int(n_samples * mal_frac)
    n_ben = n_samples - n_mal
    idx = 0
    for i in range(n_ben):
        js = gen_long_trace(malicious=False, n_events=n_events, seed=1000+i)
        fname = out_dir / f"benign_{idx:04d}.json"
        with open(fname, "w") as f: json.dump(js, f, indent=2)
        idx += 1
    for i in range(n_mal):
        js = gen_long_trace(malicious=True, n_events=n_events, seed=2000+i)
        fname = out_dir / f"malware_{idx:04d}.json"
        with open(fname, "w") as f: json.dump(js, f, indent=2)
        idx += 1
    print(f"Generated {n_samples} samples in {out_dir} (malicious={n_mal})")
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_events", type=int, default=200)
    p.add_argument("--mal_frac", type=float, default=0.5)
    args = p.parse_args()
    main(out_dir=args.out_dir, n_samples=args.n_samples, mal_frac=args.mal_frac, n_events=args.n_events)
