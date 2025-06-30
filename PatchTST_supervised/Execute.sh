OUTPUT_CSV="results_summary.csv"
echo "Ticker,MSE,MAE,RSE" > "$OUTPUT_CSV"

# Fix for np.Inf typo
sed -i 's/np.Inf/np.inf/g' /home/exouser/Extension-PatchTST/PatchTST_supervised/utils/tools.py

seq_len=336
pred_len=96
model_name=PatchTST
data_name=custom
random_seed=2021

for file in ./dataset/*.csv; do
  full_filename=$(basename "$file")
  ticker="${full_filename%%.csv}"
  model_id="${ticker}_${seq_len}_${pred_len}"

  # Rename columns before running
  python -c "
import pandas as pd
df = pd.read_csv('./dataset/${ticker}.csv')
df = df.rename(columns={'Date': 'date', 'Adj Close': 'OT'})
df.to_csv('./dataset/${ticker}.csv', index=False)
"
  echo "Running on $ticker"

  OUTPUT=$(python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path $full_filename \
    --model_id $model_id --model PatchTST \
    --data custom --features S --seq_len 96 --label_len 48 --pred_len 24 \
    --patch_len 16 --stride 8 --des 'baseline' --train_epochs 3 --batch_size 16)


  MSE=$(echo "$OUTPUT" | grep -i -oP "mse:\s*\K[0-9.]+" | tail -1)
  MAE=$(echo "$OUTPUT" | grep -i -oP "mae:\s*\K[0-9.]+" | tail -1)
  RSE=$(echo "$OUTPUT" | grep -i -oP "rse:\s*\K[0-9.]+" | tail -1)

  echo "$ticker,$MSE,$MAE,$RSE" >> "$OUTPUT_CSV"
done

echo "All runs complete. Results saved to $OUTPUT_CSV"