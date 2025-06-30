#!/usr/bin/env bash

OUTPUT_CSV="results_summary.csv"
echo "filename,MSE,MAE,RSE" > "$OUTPUT_CSV"

for file in ./dataset/*_preprocessed.csv; do
  full_filename=$(basename "$file")              # AAPL_preprocessed.csv
  ticker="${full_filename%%_preprocessed.csv}"   # Extract "AAPL"

  echo "Running on $ticker"

  # Run model and capture output
  OUTPUT=$(python run_longExp.py \
    --is_training 1 \
    --do_predict \
    --root_path ./dataset/ \
    --data_path "$full_filename" \
    --model_id "$ticker" \
    --model PatchTST \
    --data custom \
    --features S \
    --target Close \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --patch_len 13 \
    --stride 8 \
    --des batch_run \
    --train_epochs 3 \
    --batch_size 16 2>&1)

  # Extract metrics
  MSE=$(echo "$OUTPUT" | grep -oP "MSE:\s*\K[0-9.]+" | head -1)
  MAE=$(echo "$OUTPUT" | grep -oP "MAE:\s*\K[0-9.]+" | head -1)
  RSE=$(echo "$OUTPUT" | grep -oP "RSE:\s*\K[0-9.]+" | head -1)

  # Save to summary CSV
  echo "$ticker,$MSE,$MAE,$RSE" >> "$OUTPUT_CSV"
done

echo "All runs complete. Results saved to $OUTPUT_CSV"