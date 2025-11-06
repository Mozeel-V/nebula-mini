#python src/compare_attack_effects.py \
#  --base_json results/window_eval_plot/window_eval.json \
#  --attack_json results/window_eval_hotflip/window_eval.json \
#  --zero_fp_json results/window_eval_plot/zero_fp_threshold.json

python src/compare.py \
  --before results/window_eval_plot/window_eval.json \
  --after  results/window_eval/window_eval.json \
  --zero_fp results/window_eval_plot/zero_fp_threshold.json \
  --out_dir results/comparisons \
  --only_malware  # optional: only create plots for samples labeled malware

