import os

%cd /content
!git clone https://github.com/SanjayGori/Extension-PatchTST.git

%cd Extension-PatchTST/PatchTST_supervised

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
