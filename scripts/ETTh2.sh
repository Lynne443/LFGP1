
export CUDA_VISIBLE_DEVICES=0

model_name=LFGP

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96_BEST \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --c_out 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --ksh 1 \
  --type sin \
  --ksp 2 \
  --beta 0.3 \
  --freq_band_ratio 0.5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192_BEST \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --c_out 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --hidden_size 256 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --ksh 1 \
  --type sin \
  --ksp 2 \
  --beta 0.7 \
  --freq_band_ratio 0.5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336_BEST \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --c_out 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --hidden_size 128 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --ksh 1 \
  --type sin \
  --ksp 2 \
  --beta 0.3 \
  --freq_band_ratio 0.4

 
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720_BEST \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --c_out 7 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --hidden_size 128 \
  --train_epochs 15 \
  --batch_size 16 \
  --patience 10 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --ksh 1 \
  --type sin \
  --ksp 2 \
  --beta 0.3 \
  --freq_band_ratio 0.4