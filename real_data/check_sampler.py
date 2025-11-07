#!/usr/bin/env python3
from pathlib import Path
from saliency_weighted_sampler import Sampler

api_candidates = "real_data/checkpoints/api_candidates.json"
api_effect = "real_data/checkpoints/api_effectiveness.json"
sampler = Sampler(api_candidates, api_effectiveness=api_effect, benign_counts=None,
                  alpha=0.7, beta=0.25, gamma=0.05, temp=0.9)
for i in range(10):
    print(sampler.sample_api_event(i))
