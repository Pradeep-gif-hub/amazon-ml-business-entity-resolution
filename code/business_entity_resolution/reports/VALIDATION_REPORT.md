# Validation Report: Business Entity Resolution Solution

## 1. Validation Setup & Strategy

- **Methodology:** Stratified Non-Leaking 80/20 Holdout on Reference S1 Entities.
- **Random Seed:** 42
- **Validation Query Entities ($S_1$):** 19,998
- **Target Search Pool ($S_2 + S_3$):** 2,400,000 records
- **Optimal Selected Decision Threshold ($	heta$):** 0.350
- **Score Margin ($\delta$):** 0.20

## 2. Measured Validation Performance Metrics

| Metric | Value | Description |
| --- | --- | --- |
| **Macro $F_{0.5}$ (Official Metric)** | **0.8875** | Primary competition evaluation criterion |
| Macro Precision | 0.8991 | Average precision across all $S_1$ entities |
| Macro Recall | 0.8890 | Average recall across all $S_1$ entities |
| Micro $F_{0.5}$ | 0.9110 | Global link-level $F_{0.5}$ score |
| Micro Precision | 0.9481 | Global precision across all emitted links |
| Micro Recall | 0.7879 | Global recall across all ground truth links |
| Candidate Blocking Recall | 88.22% | Upper bound recall of candidate generator |
| Singleton Accuracy | 0.9493 | Accuracy on zero-match reference records |
| Non-Singleton Macro $F_{0.5}$ | 0.8392 | Macro $F_{0.5}$ on records with $\ge 1$ true match |
| US Macro $F_{0.5}$ | 0.9014 | Macro $F_{0.5}$ on US entities |
| India Macro $F_{0.5}$ | 0.8669 | Macro $F_{0.5}$ on India entities |
