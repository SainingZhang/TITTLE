import torch
from .models_ours import SelfCondDiT, QformerEncoder, Encoder, DiTiEncoder
from .sd3.mmdit import MMDiT


def SCDiT_XL2(**kwargs):
    return SelfCondDiT(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs
    )

def SCDiT_Dual_Bi_XL2(**kwargs):
    return SelfCondDiT(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16,
        dit_attention='bi', **kwargs
    )

def SCDiT_Dual_Uni_XL2(**kwargs):
    return SelfCondDiT(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16,
        dit_attention='uni', **kwargs
    )

def SCDiT_Dual_Uni0_XL2(**kwargs):
    return SelfCondDiT(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16,
        dit_attention='uni-0', **kwargs
    )

def MMDiT_XL(**kwargs):
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": kwargs['encoder_hidden_size'], "out_features": 1536},
    }
    diffusion_model = MMDiT(
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=192,
        patch_size=2,
        depth=24,
        num_patches=36864,
        adm_in_channels=kwargs['encoder_hidden_size'],
        context_embedder_config=context_embedder_config,
        device='cpu',
        dtype=torch.float,
        **kwargs
    )
    return diffusion_model

def Enc_Tiny_8(**kwargs):
    return Encoder(patch_size=8, hidden_size=256, num_heads=4, **kwargs)

def Enc_Base_8(**kwargs):
    return Encoder(patch_size=8, hidden_size=768, num_heads=12, **kwargs)

def Enc_Base_16(**kwargs):
    return Encoder(patch_size=16, hidden_size=256, num_heads=4, **kwargs)

def Enc_L_8(**kwargs):
    assert kwargs["K"] <= 24, "Enc-L/8 supports K up to 24."
    return Encoder(patch_size=8, hidden_size=768, num_heads=16, depth=24, **kwargs)

def Enc_H_8(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=768, num_heads=16, depth=32, **kwargs)

def Enc_H_8_XS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=32, **kwargs)

def Enc_H_8_XS_24(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=24, **kwargs)

def Enc_H2_8_XS(**kwargs):
    assert kwargs["K"] <= 40, "Enc-H/8 supports K up to 40."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=40, **kwargs)

def Enc_H3_8_XS(**kwargs):
    assert kwargs["K"] <= 48, "Enc-H/8 supports K up to 48."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=48, **kwargs)

def Enc_B_8_XS(**kwargs):
    assert kwargs["K"] <= 16, "Enc-B/8 supports K up to 16."
    return Encoder(patch_size=8, hidden_size=256, num_heads=16, depth=16, **kwargs)

def Enc_H_4_XS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/4 supports K up to 32."
    return Encoder(patch_size=4, hidden_size=64, num_heads=8, depth=32, **kwargs)

def Enc_B_4_XS(**kwargs):
    assert kwargs["K"] <= 16, "Enc-B/4 supports K up to 16."
    return Encoder(patch_size=4, hidden_size=64, num_heads=8, depth=16, **kwargs)

def Enc_H_8_XXS(**kwargs):
    assert kwargs["K"] <= 32, "Enc-H/8 supports K up to 32."
    return Encoder(patch_size=8, hidden_size=128, num_heads=8, depth=32, **kwargs)

# NOTE: DiTiEnc models need to specify depth
def DiTiEnc_Base_8(**kwargs):
    num_heads = 16
    assert num_heads % kwargs["K"] == 0    # K can be 1, 2, 4, 8, 16
    return DiTiEncoder(patch_size=8, hidden_size=768, num_heads=num_heads, depth=16, **kwargs)

def DiTiEnc_B_8_XS(**kwargs):
    num_heads = 16
    assert num_heads % kwargs["K"] == 0    # K can be 1, 2, 4, 8, 16
    return DiTiEncoder(patch_size=8, hidden_size=256, num_heads=num_heads, depth=16, **kwargs)

def Enc_Qformer_Bi_L_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=16, num_heads=2, depth=24,
        query_dim=16, query_heads=2, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_WL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=128, num_heads=4, depth=24,
        query_dim=128, query_heads=4, bidirectional=True, **kwargs
    )
    
def Enc_Qformer_Bi_UWL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=256, num_heads=8, depth=24,
        query_dim=256, query_heads=8, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_WL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=128, num_heads=4, depth=24,
        query_dim=128, query_heads=4, bidirectional=True, **kwargs
    )

def Enc_Qformer_Bi_UWL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=8, depth=24,
        query_dim=256, query_heads=8, bidirectional=True, **kwargs
    )

# def Enc_Qformer_Uni_L_2(**kwargs):
#     return QformerEncoder(patch_size=2, hidden_size=16, num_heads=2, depth=24,
#         query_dim=256, query_heads=8, bidirectional=False,  **kwargs
#     )

def Enc_Qformer_Uni_WL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=128, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_L_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=64, num_heads=4, depth=20,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_L2_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=64, num_heads=4, depth=24,
        query_dim=128, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=128, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=4, depth=28,
        query_dim=256, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni_WXL_2(**kwargs):
    return QformerEncoder(patch_size=2, hidden_size=256, num_heads=4, depth=28,
        query_dim=256, query_heads=4, bidirectional=False, **kwargs
    )

def Enc_Qformer_Uni0_WL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=128, num_heads=4, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, zero_init=True, **kwargs
    )

def Enc_Qformer_Uni_UWL_1(**kwargs):
    return QformerEncoder(patch_size=1, hidden_size=256, num_heads=8, depth=24,
        query_dim=256, query_heads=8, bidirectional=False, **kwargs
    )

DiT_models = {
    'SCDiT-XL/2': SCDiT_XL2,
    'SCDiT-Dual-Bi-XL/2': SCDiT_Dual_Bi_XL2,
    'SCDiT-Dual-Uni-XL/2': SCDiT_Dual_Uni_XL2,
    'SCDiT-Dual-Uni0-XL/2': SCDiT_Dual_Uni0_XL2,
    'MMDiT_XL': MMDiT_XL,
}

Enc_models = {
    'Enc-Tiny/8': Enc_Tiny_8,
    'Enc-Base/8': Enc_Base_8,
    'Enc-L/8': Enc_L_8,
    'Enc-H/8': Enc_H_8,
    'Enc-H/8-XS': Enc_H_8_XS,
    'Enc-H/8-XS-24': Enc_H_8_XS_24,
    'Enc-H2/8-XS': Enc_H2_8_XS,
    'Enc-H3/8-XS': Enc_H3_8_XS,
    'Enc-B/8-XS': Enc_B_8_XS,
    'Enc-H/4-XS': Enc_H_4_XS,
    'Enc-B/4-XS': Enc_B_4_XS,
    'Enc-H/8-XXS': Enc_H_8_XXS,
    'Enc-Base/16': Enc_Base_16,
    'DiTiEnc-B/8': DiTiEnc_Base_8,
    'DiTiEnc-B/8-XS': DiTiEnc_B_8_XS,
    'Enc-Qformer-Bi-L/2': Enc_Qformer_Bi_L_2,
    'Enc-Qformer-Bi-WL/2': Enc_Qformer_Bi_WL_2,
    'Enc-Qformer-Bi-UWL/2': Enc_Qformer_Bi_UWL_2,
    'Enc-Qformer-Bi-WL/1': Enc_Qformer_Bi_WL_1,
    'Enc-Qformer-Bi-UWL/1': Enc_Qformer_Bi_UWL_1,
    'Enc-Qformer-Uni-L/2': Enc_Qformer_Uni_L_2,
    'Enc-Qformer-Uni-L2/2': Enc_Qformer_Uni_L2_2,
    'Enc-Qformer-Uni-WL/2': Enc_Qformer_Uni_WL_2,
    'Enc-Qformer-Uni-WL/1': Enc_Qformer_Uni_WL_1,
    'Enc-Qformer-Uni-WXL/1': Enc_Qformer_Uni_WXL_1,
    'Enc-Qformer-Uni-WXL/2': Enc_Qformer_Uni_WXL_2,
    'Enc-Qformer-Uni0-WL/1': Enc_Qformer_Uni0_WL_1,
    'Enc-Qformer-Uni-UWL/1': Enc_Qformer_Uni_UWL_1,
}

selftok_ckpts = {
    'v0.0': "",
    'v2.6': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v2.6_n/2024-08-06_time_18_30_51/output/ckpt/iter_409999.pth",
    'v3.3': "",
    'v4.2.6': "s3://bucket-9122-wulan/outputs/l00574761/selftok/encoder_v2.6_sd3/2024-08-08_time_21_45_00/output/configs/mimo/selftok/encoder/v2.6_sd3.yml/27570833000.0/ckpt/iter_99999.pth",
    'v4.2.6-op-mix-aspect': "s3://bucket-9122-wulan/outputs/l00574761/selftok/encoder_v4.2.6-forcerecon-open-mres/2024-08-20_time_09_33_00/output/ckpt/iter_109999.pth",
    'v4.2.15-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.15-op/2024-08-21_time_18_30_51/output/ckpt/iter_79999.pth",
    'v4.2.16-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.16-op/2024-08-21_time_18_30_51/output/ckpt/iter_79999.pth",
    'v4.2.17-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.17-op/2024-08-22_time_11_30_51/output/ckpt/iter_49999.pth",
    'v4.0.0': "",
    'v0.0-op': "",
    'v2.6-op': "",
    'v4.2.6-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.6-forcerecon-open/2024-08-14_time_11_30_51/output/ckpt/iter_139999.pth",
    'v4.2.6-img': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.6-img/2024-08-21_time_18_30_51/output/ckpt/iter_109999.pth",
    'v4.2.12-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.12-forcerecon-open/2024-08-14_time_18_30_51/output/ckpt/iter_119999.pth",
    'v4.2.6-vmode-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.6-open-vmode/2024-08-14_time_12_30_51/output/ckpt/iter_119999.pth",
    'v4.2.16-op': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.16-op/2024-08-21_time_18_30_51/output/ckpt/iter_109999.pth",
    'v4.2.18-old': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.18/2024-08-26_time_18_30_51/output/ckpt/iter_79999.pth",
    'v4.0.0-op': "",
    '480-inet': "s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.18-fixtime-adp-lognorm/2024-08-30_time_18_30_51/output/ckpt/iter_109999.pth",
    '480-1M': "s3://bucket-9122-wulan/outputs/l00574761/selftok/encoder_v4-2-18-1mdata/2024-08-31_time_11_22_00/output/ckpt/iter_59999.pth",
    '680-inet': "",
}