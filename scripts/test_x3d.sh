MODEL_NAME=x3d

nvidia-smi
CUDA_VISIBLE_DEVICES=3 \
python3 test.py \
    --output_dir output \
    --load_dir output/x3d/number \
    --use_metrics mse+js+balanced_acc \
    --data_dir dataset \
    --num_frame 16 \
    --model_name $MODEL_NAME \
    --batch_size 16 \