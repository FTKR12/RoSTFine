MODEL_NAME=rostfine

nvidia-smi
CUDA_VISIBLE_DEVICES=3 \
python3 test.py \
    --output_dir output \
    --load_dir output/rostfine/number \
    --use_metrics mse+js+balanced_acc \
    --data_dir dataset \
    --num_frame 8 \
    --model_name $MODEL_NAME \
    --use_div gs+gt+st \
    --use_feat vg+vs+vt \
    --alpha 1.0 \
    --topk 3 \
    --batch_size 16 \