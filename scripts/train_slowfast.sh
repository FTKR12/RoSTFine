MODEL_NAME=slowfast

nvidia-smi
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
    --output_dir output \
    --data_dir dataset \
    --num_frame 32 \
    --model_name $MODEL_NAME \
    --batch_size 4 \
    --epochs 3 \
    --optim SGD \
    --lr 0.005