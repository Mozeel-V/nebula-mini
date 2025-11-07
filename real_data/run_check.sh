python - <<'PY'
from real_data.saliency_weighted_sampler import Sampler
s = Sampler("checkpoints/api_candidates.json", "checkpoints/api_effectiveness.json", None, temp=0.9)
for i in range(5):
    print(s.sample_api_event(i))
PY