# Experiment Log: Entity Resolution Model Iterations

| Exp # | Approach / Model | Blocking Recall | Macro Prec | Macro Rec | Macro F0.5 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline Heuristic Rule Matcher | 84.95% | 0.8120 | 0.7640 | 0.8018 | Initial exact/fuzzy rule baseline |
| 2 | Multi-Key Inverted Index Blocking | 89.72% | 0.8410 | 0.8120 | 0.8350 | Added nospace domain & address keys |
| 3 | Full GBDT + Vectorized Batched Scoring + Address Veto | 88.22% | 0.8991 | 0.8890 | **0.8875** | Optimal threshold 0.350 + address veto |
