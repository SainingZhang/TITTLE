# VQGAN-LC-SD3: What makes good image quantization

## 🔧 Preparation

### Prepare Environment
```bash
pip install -r requirements.txt
```

### Prepare Datasets

Download ImageNet1K dataset and arranged with the following layout:

```
├── /ImageNet1K/
│  ├── /train/
│  ├──  ├── n01440764
│  ├──  ├── n01443537
│  ├──  ├── .........
│  ├── /val/
│  ├──  ├── n01440764
│  ├──  ├── n01440764
│  ├──  ├── .........
```

Download the train/val split of ImageNet1K from our [Google Drive](https://drive.google.com/drive/folders/11mxqPcm8IbbcD6F6DUjufOxcQIXucBcT?usp=sharing).


## 🚗 Runing

### Image Quantization

#### Initialized Codebook Generation

The Initialized codebook should be first downloaded from our [Google Drive](https://drive.google.com/drive/folders/1eTKbOoI8ootxexNgBLs0Dvz-qOdZM21m?usp=sharing) or generate with the following script:
```bash
imagenet_path="IMAGENET PATH"
cd codebook_generation
sh run.sh
```

#### VQGAN-LC Training
Training VQGAN-LC with a codebook size 100K with the following script:

```bash
cd vqgan-gpt-lc
imagenet_path="IMAGENET PATH"
codebook_path="INIT CODEBOOK PATH"
torchrun --nproc_per_node 8 training_vqgan.py \
    --batch_size 256 \
    --image_size 256 \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 5e-4 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --num_workers 16 \
    --vq_config_path vqgan_configs/vq-f16.yaml \
    --output_dir "train_logs_vq/vqgan_lc_100K" \
    --log_dir "train_logs_vq/vqgan_lc_100K" \
    --disc_start 50000 \
    --n_vision_words 100000 \
    --local_embedding_path $codebook_path \
    --tuning_codebook 0 \
    --use_cblinear 1 \
    --embed_dim 8
```

We provide VQGAN-LC-100K trained for 20 epoches at [Google Drive](https://drive.google.com/drive/folders/12824gtaR_upGH1DJRNfAjQFiw8c1FmwJ?usp=sharing).

#### VQGAN-LC Testing
Testing VQGAN-LC for image quantization with the following script:

```bash
cd vqgan-gpt-lc
imagenet_path="IMAGENET PATH"
codebook_path="INIT CODEBOOK PATH"
vq_path="VQGAN-LC PATH"
torchrun --nproc_per_node 1 eval_reconstruction.py \
        --batch_size 8 \
        --image_size 256 \
        --lr 9e-3 \
        --n_class 1000 \
        --imagenet_path $imagenet_path \
        --vq_config_path vqgan_configs/vq-f16.yaml \
        --output_dir "log_eval_recons/vqgan_lc_100K_f16" \
        --log_dir "log_eval_recons/vqgan_lc_100K_f16" \
        --quantizer_type "org" \
        --local_embedding_path $codebook_path \
        --stage_1_ckpt $vq_path \
        --tuning_codebook 0 \
        --embed_dim 8 \
        --n_vision_words 100000 \
        --use_cblinear 1 \
        --dataset "imagenet"
```

PSNR and SSIM are computed by [pyiqa](https://github.com/chaofengc/IQA-PyTorch). rFID is calculated by [cleanfid](https://github.com/GaParmar/clean-fid).

## 📏 Checkpoints

### Image Quantization
| Method  | Resolution | Utilization Rate | rFID Score | Checkpoints |
|---------|---------------|----------|----------|----------|
| VQGAN-LC | f16 | 99.9%     | 2.62 | [Google Drive](https://drive.google.com/drive/folders/12824gtaR_upGH1DJRNfAjQFiw8c1FmwJ?usp=sharing)
| VQGAN-LC | f8 | 99.5%     | 1.29 | [Google Drive](https://drive.google.com/drive/folders/12824gtaR_upGH1DJRNfAjQFiw8c1FmwJ?usp=sharing)


## 👨‍🏫 Acknowledgement

The evaluation tools are used from [pyiqa](https://github.com/chaofengc/IQA-PyTorch) and [cleanfid](https://github.com/GaParmar/clean-fid).

