MODEL_NAME=rostfine

nvidia-smi
CUDA_VISIBLE_DEVICES=2 \
python3 train.py \
    --output_dir output \
    --data_dir dataset \
    --num_frame 8 \
    --model_name $MODEL_NAME \
    --use_div gs+gt+st \
    --use_feat vg+vs+vt \
    --alpha 1.0 \
    --topk 3 \
    --batch_size 8 \
    --epochs 300 \
    --optim SGD \
    --lr 0.005