# 📈 Adaptive Time-Aware PatchTST – Project Roadmap (Team 7)

This project enhances the PatchTST model for financial time series forecasting using adaptive patching, time-aware embeddings, and multivariate indicators. Below is the step-by-step blueprint our team will follow.

---

## 📚 Table of Contents

0. [0. Kick-off](#-0-kick-off-1-day)
1. [1. Data Pipeline](#-1-data-pipeline-3-days)
2. [2. Adaptive Windowing](#-2-adaptive-windowing-4-days)
3. [3. Dynamic Patching](#%EF%B8%8F-3-dynamic-patching-5-days)
4. [4. Time-Aware Positional Embedding](#-4-time-aware-positional-embedding-3-days)
5. [5. Multivariate Feature Handler](#-5-multivariate-feature-handler-2-days)
6. [6. Training & Experimentation](#-6-training--experimentation-7-days)
7. [7. Reporting & Visualization](#-7-reporting--visualization-3-days)
8. [8. Contingency & Plan B](#-8-contingency--plan-b-1-day)
9. [9. Final Polish & Submission](#-9-final-polish--submission-2-days)
10. [Final Decisions Needed](#final-decisions-needed)

---

## ✅ 0. Kick-off (1 day)

| Task                                                                                                                                            | Goal                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **0.1 Quick Sync** – 30-min call: confirm objective, extensions (Dynamic Patching, Time-Aware Embedding, Adaptive Windowing), deadlines, roles. | Align everyone and lock scope |
| **0.2 Repo & Boards** – create / tidy GitHub repo, one ClickUp/Jira board with the same phase names as below.                                   | Organize project assets     |
| **0.3 Baseline Check** – run the *exact* PatchTST weather benchmark command and store logs + metrics in `/baseline/`.                          | Baseline reproducibility    |

✅ **Done when**: repo + board exist and baseline metrics are committed.

---

## 🛠 1. Data Pipeline (3 days)

| Task                                                                                                                             | Plain-English Goal          |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **1.1 Raw Grab** – download Kaggle S&P 500 OHLCV file to `/data/raw/`.                                                           | Source historical data      |
| **1.2 Cleaning Script** – fill blanks, convert `Date` to `datetime`, keep only required columns.                                 | Ensure clean, usable data   |
| **1.3 Holiday & Gap Flags** – add `is_gap` (when `Date_i – Date_{i-1} > 1 day`).                | Tag weekends/holidays       |
| **1.4 Derived Indicators** – compute RSI, SMA, EMA, MACD, ATR, daily log-return. Include formulas as comments.                   | Create technical signals    |
| **1.5 10-Row Snapshot** – export a CSV with 10 representative rows.                                                              | Quick visual reference      |

✅ **Done when**: `data/processed/train.csv` exists and the 10-row sample is committed.

---

## 🔄 2. Adaptive Windowing (4 days)

| Task                                                                                                           | How to explain to anyone               |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **2.1 Market Regime Finder** – calculate 5-day rolling volatility of log-returns.                              | Track market turbulence                |
| **2.2 Volatility Normalization** – Normalize to [0, 1] and store as `volatility_norm`.                         | Makes volatility scale consistent      |
| **2.3 Window Logic** – Map volatility → window size (e.g., low vol = 50, high vol = 10). Store as `window_len`.| Dynamic patch control                  |
| **2.4 Adaptive Patch** – Round each `window_len` to nearest multiple of 8.                                     | Stability for transformer batching     |
| **2.5 Unit Test** – On toy data, confirm high volatility shrinks the window.                                   | Proves window logic works              |
| **2.6 Notebook Demo** – Plot volatility vs. chosen window length; label axes.                                  | Visual explanation                     |

✅ **Done when**: plot clearly shows adaptive shrinking/growing windows and tests pass.

---

## ⚙️ 3. Dynamic Patching (5 days)

| Task                                                                                                                    | Layman description                                |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **3.1 Patch Generator Refactor** – replace fixed `patch_len` with `patch_len = window_len // k` (start with `k=3`).     | Patch size adapts to window                       |
| **3.2 Gap-Aware Slicer** – skip patches that cross holidays or weekends.                                                | Prevents broken input                             |
| **3.3 Patch Metadata** – record each patch’s `start_date`, `end_date`, and `real_len`.                                  | Easier debugging and visualization                |
| **3.4 Distribution Check** – plot histogram of patch lengths; no zeros, reasonable range.                              | Sanity check on patch generator                   |
| **3.5 Ablation Toggle** – add `--dynamic_patching false` flag to fallback to static logic.                             | For fair comparison and backup                    |

✅ **Done when**: histogram looks correct and CLI toggle works end-to-end.

---

## 🧭 4. Time-Aware Positional Embedding (3 days)

| Task                                                                                                                 | Explanation                          |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **4.1 Δt Vector** – compute normalized time gap (days) between each row.                                            | Captures irregularity in timestamps  |
| **4.2 Embedding Layer** – Add sinusoidal or learned embedding for Δt.                                               | Time gaps as transformer tokens      |
| **4.3 Integration Test** – Overfit on 100 rows; loss must decrease.                                                 | Proves the code is wired correctly   |

✅ **Done when**: model trains without error and `model.summary()` shows time-aware embedding layer.

---

## 📊 5. Multivariate Feature Handler (2 days)

| Task                                                                                                                             | Goal                                  |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **5.1 Feature Stack** – combine OHLCV + indicators into one tensor `[batch, seq, channels]`.                                     | Multivariate forecasting              |
| **5.2 Cross-Channel Attention** – apply joint attention (concatenate channels before patching).                                  | Let model learn cross-feature dynamics|

✅ **Done when**: forward pass works with 10+ input channels.

---

## 🔁 6. Training & Experimentation (7 days)

| Step                                                                                                                  | What happens                          |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **6.1 Lightning Trainer** – convert training loop to PyTorch Lightning.                                               | Cleaner logs, callbacks, checkpoints  |
| **6.2 First Full Run** – train Enhanced PatchTST on 10 tickers (AAPL, MSFT, etc.).                                    | Baseline performance on target data   |
| **6.3 Hyper-Param Grid** – search `d_model`, `n_heads`, `patch_len divisor k`, learning rate.                         | Optimize model performance            |
| **6.4 Ablations** – train 3 variants: (a) no Δt embedding, (b) no dynamic patch, (c) no adaptive window.              | Prove each module's importance        |
| **6.5 Comparison Plots** – RMSE bar chart for each variant + original PatchTST.                                       | Visualize the impact of improvements  |

✅ **Done when**: results + plots are in `/results/phase6/` and README explains model wins/losses.

---

## 📝 7. Reporting & Visualization (3 days)

| Deliverable                                                                                                                      | Checklist                            |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **7.1 Architecture Diagram** – block diagram: input → windowing → dynamic patches → transformer → forecast.                     | System view for report                |
| **7.2 Graph Annotations** – each plot must include title, axis labels with units, and caption.                                  | Scientific visualization              |
| **7.3 Layman Write-up** – a 2-page jargon-free summary of key ideas.                                                            | For final presentation/report         |
| **7.4 Appendix** – include indicator formulas, 10-row data sample, and hardware used.                                            | Completeness                          |

✅ **Done when**: Word draft is ready and plots are publication-ready.

---

## 🧯 8. Contingency & Plan B (1 day)

| Task                                                                                                          | Purpose                      |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **8.1 Fallback Config** – original PatchTST + static patch_len only.                                          | Baseline safety net          |
| **8.2 Rollback Script** – `run_baseline.sh` gets baseline metrics in <1 hr on Colab.                          | Reproducibility              |
| **8.3 Decision Matrix** – If Enhanced RMSE ≤ baseline × 1.05 → keep; else fallback in final report.           | Clear judgment rule          |

✅ **Done when**: fallback run finishes and matrix is added to the repo.

---

## 🧹 9. Final Polish & Submission (2 days)

1. Freeze code (`git tag v1.0-final`)
2. Ensure `pytest` passes all unit tests
3. Zip `src/`, `results/`, and report → submit to LMS/GitHub
4. Deliver 15-slide deck + 5-min demo video

✅ **Done when**: confirmation email is received for successful submission.

---

## ❗ Final Decisions Needed
# 🧠 PatchTST Extension for Multivariate and Irregular Time Series Forecasting

This project extends the [PatchTST](https://github.com/yuqinie98/PatchTST) model for more realistic financial time series data. We add new capabilities to make it robust for multivariate inputs, irregular time intervals, and adaptive patching needs.

---

## 🔧 Enhancements Implemented

### 1. **Dynamic Patching**
- **Original:** Patch length was fixed (e.g., 16) across all runs.
- **Now:** Patch length is dynamically determined based on a window-to-patch ratio (`window_len // k`).
- **Why:** Financial data may not align to fixed time slices due to holidays/gaps. Dynamic patching adapts automatically.
- **How it helps:** Prevents slicing through gaps; ensures better learning with cleaner patch boundaries.

---

### 2. **Adaptive Flatten Head**
- **Original:** Flatten head assumed a fixed patch count.
- **Now:** Flatten head is rebuilt dynamically based on actual number of patches.
- **Why:** Needed to work with variable patch sizes.
- **How it helps:** Keeps model flexible and avoids size mismatch errors.

---

### 3. **Dynamic Positional Encoding**
- **Original:** One fixed positional encoding tensor for fixed patch count.
- **Now:** Positional encodings are cached and reused dynamically based on patch count.
- **Why:** Essential for dynamic patches.
- **How it helps:** Enables correct temporal understanding even with changing input shape.

---

### 4. **Time-Aware Embedding**
- **New Addition:** Positional embeddings now include actual time gaps (delta time).
- **Why:** In real data, time intervals may not be uniform.
- **How it helps:** Model is aware of missing days/weekends and learns time dynamics better.

---

### 5. **CLI Support for Dynamic Patch Toggle**
- **New:** A command-line flag `--use_dynamic_patch` to enable or disable dynamic patching.
- **Why:** Easy to switch between baseline and enhanced model.
- **How it helps:** Simplifies ablation studies and debugging.

---

## 📁 Directory Overview

- `PatchTST_supervised/` — Core model and training logic
- `data/` — Input data files (e.g., `weather.csv`, `AAPL.csv`)
- `scripts/derived.py` — Code for generating technical indicators

---

## 🚀 Run Example

Baseline (fixed patching):
```bash
import pandas as pd

df = pd.read_csv("dataset/AAPL.csv")
df.rename(columns={"Date": "date"}, inplace=True)
df.to_csv(f'dataset/AAPL.csv', index=False)
!python run_longExp.py \
  --is_training 1 \
  --do_predict \
  --root_path ./dataset/ --data_path AAPL.csv \
  --model_id PatchTST_aapl \
  --model PatchTST \
  --data custom --features S --target Close \
  --seq_len 96 --label_len 48 --pred_len 24 \
  --patch_len 13 --stride 8 \
  --des dyn13_test \
  --train_epochs 3 \
  --batch_size 16
```
---
# THE END
