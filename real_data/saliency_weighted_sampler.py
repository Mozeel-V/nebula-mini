#!/usr/bin/env python3
"""
Saliency-weighted API sampler for Nebula attacks.

Combines three signals into a probability distribution over API candidates:
  1. Effectiveness score (mean Δ in malware score from compute_api_effectiveness.py)
  2. Frequency prior (how often the API occurs in benign traces)
  3. Uniform exploration term (γ) to ensure diversity

The resulting weighted sampler can be used for both saliency-based replacement
and GA mutation operators to replace/insert realistic benign API calls.

Example usage:
    from saliency_weighted_sampler import Sampler
    samp = Sampler(
        api_candidates="checkpoints/api_candidates.json",
        api_effectiveness="checkpoints/api_effectiveness.json",
        benign_counts=None,  # optional JSON of benign API frequencies
        alpha=0.7, beta=0.25, gamma=0.05, temp=1.0
    )
    new_event = samp.sample_api_event(idx=0)
"""

import json
import random
import numpy as np
from pathlib import Path


def load_counts(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class Sampler:
    def __init__(
        self,
        api_candidates: str,
        api_effectiveness: str = None,
        benign_counts: str = None,
        alpha: float = 0.7,
        beta: float = 0.25,
        gamma: float = 0.05,
        temp: float = 1.0,
    ):
        """
        api_candidates: JSON list like ["api:ReadFile", "api:connect", ...]
        api_effectiveness: JSON list of {"api": "api:ReadFile", "mean_delta": 0.12, "count": 50}
        benign_counts: optional JSON map {"api:ReadFile": 123, "api:connect": 77}
        alpha, beta, gamma: weight coefficients
        temp: temperature for sampling sharpness (<1 = exploitative, >1 = exploratory)
        """
        self.apis = json.load(open(api_candidates, "r", encoding="utf-8"))

        # Load effectiveness data (if available)
        eff_map = {}
        if api_effectiveness and Path(api_effectiveness).exists():
            eff_list = json.load(open(api_effectiveness, "r", encoding="utf-8"))
            for d in eff_list:
                eff_map[d["api"]] = float(d.get("mean_delta", 0.0))

        # Load benign frequency counts (optional)
        freq_map = load_counts(benign_counts) if benign_counts else {}

        # Build numeric arrays
        eff_vec = np.array([eff_map.get(a, 0.0) for a in self.apis], dtype=float)
        freq_vec = np.array([freq_map.get(a, 0.0) for a in self.apis], dtype=float)

        # Normalize both to [0, 1]
        eff_norm = (eff_vec - eff_vec.min()) / (eff_vec.max() - eff_vec.min() + 1e-12)
        freq_norm = (freq_vec - freq_vec.min()) / (freq_vec.max() - freq_vec.min() + 1e-12)

        # Combine weights and apply temperature scaling
        weights = alpha * eff_norm + beta * freq_norm + gamma
        if temp != 1.0:
            weights = np.exp(np.log(weights + 1e-12) / temp)

        # Normalize to get probabilities
        weights = np.clip(weights, 0.0, None)
        if weights.sum() == 0:
            probs = np.ones(len(self.apis)) / len(self.apis)
        else:
            probs = weights / weights.sum()

        self.probs = probs.tolist()

    # ---------------------------------------------------------------
    # Sampling helpers
    # ---------------------------------------------------------------

    def sample_api(self) -> str:
        """Return a random API token (e.g., 'api:ReadFile') weighted by learned probabilities."""
        return random.choices(self.apis, weights=self.probs, k=1)[0]

    def sample_event_from_api(self, api_token: str, idx: int = 0) -> str:
        """Generate a realistic event string for a given API name."""
        api = api_token.split("api:")[-1]
        low = api.lower()

        # Simple contextual augmentation
        if low in ("readfile", "createfilew", "writefile"):
            return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"
        if low in ("connect", "send", "recv"):
            return f"api:{api} ip:127.0.0.1"
        if low.startswith("reg"):
            return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"
        if "process" in low:
            return f"api:{api} pid:{1000 + idx}"
        if "thread" in low:
            return f"api:{api} tid:{2000 + idx}"
        return f"api:{api}"

    def sample_api_event(self, idx: int = 0) -> str:
        """Directly return a sampled API event string."""
        api = self.sample_api()
        return self.sample_event_from_api(api, idx)
