#!/usr/bin/env bash

OUTPUT_CSV="results_summary.csv"
echo "MSE,MAE,RSE" > "$OUTPUT_CSV"

seq_len=336
pred_len=720
model_name=PatchTST
data_name=custom
random_seed=2021

for file in ./dataset/*.csv; do
  full_filename=$(basename "$file")
  ticker="${full_filename%%.csv}"
  model_id="${ticker}_${seq_len}_${pred_len}"

  echo "Running on $ticker"

  OUTPUT=$(python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --do_predict \
    --root_path ./dataset/ \
    --data_path "$full_filename" \
    --model_id "$model_id" \
    --model $model_name \
    --data $data_name \
    --features S \
    --target OT \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --itr 1 \
    --batch_size 128 \
    --use_amp \
    --use_gpu True \
    --learning_rate 0.0001 2>&1)


  MSE=$(echo "$OUTPUT" | grep -i -oP "mse:\s*\K[0-9.]+" | tail -1)
  MAE=$(echo "$OUTPUT" | grep -i -oP "mae:\s*\K[0-9.]+" | tail -1)
  RSE=$(echo "$OUTPUT" | grep -i -oP "rse:\s*\K[0-9.]+" | tail -1)

  echo "$MSE,$MAE,$RSE,$OUTPUT" >> "$OUTPUT_CSV"
done

echo "All runs complete. Results saved to $OUTPUT_CSV"