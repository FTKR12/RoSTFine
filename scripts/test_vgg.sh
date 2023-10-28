MODEL_NAME=vgg

nvidia-smi
CUDA_VISIBLE_DEVICES=2 \
python3 test.py \
    --output_dir output \
    --load_dir output/vgg/number \
    --use_metrics mse+js+balanced_acc \
    --data_dir dataset \
    --model_name $MODEL_NAME \
    --batch_size 16 \