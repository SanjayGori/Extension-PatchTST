**Project Road Map – “Adaptive Time-Aware PatchTST” (Team 7)**
This is a step-by-step blueprint you can follow exactly as written. Each phase ends with a clear “Done ✓” milestone so everyone knows when to move on.

---

### 0. Kick-off (1 day)

| Why?                                                                                                                                            | To align everyone and lock scope |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **0.1 Quick Sync** – 30-min call: confirm objective, extensions (Dynamic Patching, Time-Aware Embedding, Adaptive Windowing), deadlines, roles. |                                  |
| **0.2 Repo & Boards** – create / tidy GitHub repo, one ClickUp/Jira board with the same phase names as below.                                   |                                  |
| **0.3 Baseline Check** – run the *exact* PatchTST weather benchmark command (already done) and store logs + metrics in `/baseline/`.            |                                  |

**Done ✓** when repo + board exist and baseline metrics are committed.

---

### 1. Data Pipeline (3 days)

| Task                                                                                                                             | Plain-English Goal |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **1.1 Raw Grab** – download Kaggle S\&P 500 OHLCV file to `/data/raw/`.                                                          |                    |
| **1.2 Cleaning Script** – fill obvious blanks, convert `Date` to `datetime`, keep only required columns.                         |                    |
| **1.3 Holiday & Gap Flags** – add `is_holiday` (NYSE calendar) and `is_gap` (when `Date_i – Date_{i-1} > 1 day`).                |                    |
| **1.4 Derived Indicators** – compute RSI, SMA, EMA, MACD, ATR, daily log-return. Save math formulas as comments in the notebook. |                    |
| **1.5 10-Row Snapshot** – export a CSV with 10 representative rows; this will be pasted into every future report.                |                    |

**Done ✓** when `data/processed/train.csv` exists and the 10-row sample is committed.

---

### 2. Adaptive Windowing (4 days)

| Task                                                                                                           | How to explain to anyone |
| -------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **2.1 Market Regime Finder** – calculate 5-day rolling volatility of log-returns.                              |                          |
| **2.2 Window Logic** – map volatility → window size (low vol = 50, high vol = 10). Store `window_len` per row. |                          |
| **2.3 Unit Test** – for a hand-made series, assert that high volatility shrinks the window.                    |                          |
| **2.4 Notebook Demo** – plot volatility vs. chosen window length for one ticker; label axes and units.         |                          |

**Done ✓** when a plot clearly shows window shrinking/growing with volatility and tests pass.

---

### 3. Dynamic Patching (5 days)

| Task                                                                                                                    | Layman description |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **3.1 Patch Generator Refactor** – replace fixed `patch_len` with `patch_len = window_len // k` (choose `k = 3` first). |                    |
| **3.2 Gap-Aware Slicer** – ensure patches never slice across holidays/weekends; skip gaps entirely.                     |                    |
| **3.3 Patch Metadata** – record each patch’s **start\_date**, **end\_date**, **real\_len** so we can debug.             |                    |
| **3.4 Distribution Check** – histogram of patch lengths; verify no zeros and reasonable spread.                         |                    |
| **3.5 Ablation Toggle** – CLI flag `--dynamic_patching false` to fall back to original logic (contingency).             |                    |

**Done ✓** when histogram looks right and the toggle works.

---

### 4. Time-Aware Positional Embedding (3 days)

| Task                                                                                                                 | Explanation |
| -------------------------------------------------------------------------------------------------------------------- | ----------- |
| **4.1 Δt Vector** – compute time gap (days) between successive rows; normalize to 0–1.                               |             |
| **4.2 Embedding Layer** – add *either* sinusoidal or learned embedding for `Δt`; pick whichever trains faster first. |             |
| **4.3 Integration Test** – overfit on 100 rows and confirm loss goes down (proves wiring correct).                   |             |

**Done ✓** when model trains without errors and `model.summary()` shows the new embedding layer.

---

### 5. Multivariate Feature Handler (2 days)

| Task                                                                                                                                  | Goal |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| **5.1 Feature Stack** – combine OHLCV + indicators into one tensor `[batch, seq, channels]`.                                          |      |
| **5.2 Cross-Channel Attention** – switch PatchTST to **joint** attention (not channel-wise) by concatenating channels before patches. |      |

**Done ✓** when a single forward pass works with >10 channels.

---

### 6. Training & Experimentation (7 days)

| Step                                                                                                                  | What happens |
| --------------------------------------------------------------------------------------------------------------------- | ------------ |
| **6.1 Lightning Trainer** – migrate training loop to PyTorch Lightning for clean logs & checkpointing.                |              |
| **6.2 First Full Run** – train Enhanced PatchTST on 10 tickers (AAPL…); log RMSE/MAE.                                 |              |
| **6.3 Hyper-Param Grid** – search `d_model`, `n_heads`, `patch_len divisor k`, learning rate.                         |              |
| **6.4 Ablations** – train three variants: **a)** no Δt embedding, **b)** no dynamic patch, **c)** no adaptive window. |              |
| **6.5 Comparison Plots** – bar chart of RMSE across variants; include original PatchTST baseline.                     |              |

**Done ✓** when results & plots are pushed to `/results/phase6/` and README explains wins/losses.

---

### 7. Reporting & Visualization (3 days)

| Deliverable                                                                                                                      | Checklist |
| -------------------------------------------------------------------------------------------------------------------------------- | --------- |
| **7.1 Architecture Diagram** – update the coloured block diagram (input → windowing → dynamic patches → transformer → forecast). |           |
| **7.2 Graph Annotations** – each figure gets: title, X/Y labels with units, 1-line caption.                                      |           |
| **7.3 Layman Write-up** – a 2-page summary explaining *why* each extension matters, no jargon.                                   |           |
| **7.4 Appendix** – include formulas, 10-row data sample, hardware specs table.                                                   |           |

**Done ✓** when Word doc draft is ready for faculty review.

---

### 8. Contingency & Plan B (1 day)

| What if results disappoint?                                                                                           |
| --------------------------------------------------------------------------------------------------------------------- |
| **8.1 Fallback Config** – original PatchTST + simple feature scaling (acts as base).                                  |
| **8.2 Rollback Script** – bash script `run_baseline.sh` reproducibly gets baseline numbers in <1 h on Colab.          |
| **8.3 Decision Matrix** – if Enhanced RMSE ≤ baseline × 1.05 by July 5 → keep; else present fallback in final report. |

**Done ✓** when fallback run completes and matrix is in repo.

---

### 9. Final Polish & Submission (2 days)

1. Freeze code (`git tag v1.0-final`).
2. Run `pytest` – all unit tests green.
3. Zip `src/`, `results/`, report and push to LMS/GitHub.
4. 15-slide deck + 5-min demo video.

**Done ✓** when upload confirmation is received.

---

## What You Might Still Need

* **Exact hardware** you plan to train on (JS2? Colab Pro? local GPU) – affects training schedule.
* **Choice of hyper-param tuner** (manual grid vs. Optuna).
* **List of final tickers** (all S\&P 500 or top 50?).

Let me know if any of these details are undecided, and I’ll slot them into the roadmap.
