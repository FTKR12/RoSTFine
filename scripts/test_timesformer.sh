MODEL_NAME=timesformer

nvidia-smi
CUDA_VISIBLE_DEVICES=3 \
python3 test.py \
    --output_dir output \
    --load_dir output/timesformer/number \
    --use_metrics mse+js+balanced_acc \
    --data_dir dataset \
    --num_frame 8 \
    --model_name $MODEL_NAME \
    --batch_size 16 \