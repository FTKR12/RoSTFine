MODEL_NAME=x3d

nvidia-smi
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
    --output_dir output \
    --data_dir dataset \
    --num_frame 16 \
    --model_name $MODEL_NAME \
    --batch_size 8 \
    --epochs 300 \
    --optim SGD \
    --lr 0.005