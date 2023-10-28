MODEL_NAME=vgg

nvidia-smi
CUDA_VISIBLE_DEVICES=3 \
python3 train.py \
    --output_dir output \
    --data_dir dataset \
    --model_name $MODEL_NAME \
    --batch_size 8 \
    --epochs 300 \
    --optim SGD \
    --lr 0.005