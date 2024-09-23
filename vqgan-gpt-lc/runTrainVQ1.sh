#imagenet_path=""
imagenet_path="/cache/imagenet1k/"
codebook_path="/cache/VQGAN-LC/codebook-100K.pth"

####Training VQGAN-LC with the generated codebook
torchrun --nproc_per_node 4 --master_port=13247 training_vqgan.py \
    --batch_size 100 \
    --image_size 256 \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 1e-4 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --num_workers 16 \
    --vq_config_path vqgan_configs/vq-f16.yaml \
    --output_dir "/cache/train_logs_vq/vqgan_lc_100K_0919_2layer" \
    --log_dir "/cache/train_logs_vq/vqgan_lc_100K_0919_2layer" \
    --disc_start 50000 \
    --n_vision_words 100000 \
    --local_embedding_path $codebook_path \
    --quantizer_type "org" \
    --tuning_codebook 0 \
    --use_cblinear 1 \
    --embed_dim 8