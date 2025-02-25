encoder_channel_list = [768, 384, 192, 96]  # for swin
decoder_channel_list = encoder_channel_list

import yaml
from util.opts import dict2namespace
from models.diff_model import VideoSaliencyModel
from models.mvit import MViT
from models.saliency_decoder.sal_unet import SalUNet
from datasets.dhf1k_data import DHF1KDatasetMultiFrames
from datasets.holly2wood_dataset import HollyDataset
from datasets.ucf_dataset import UCFDataset
from datasets.memor_dataset import MEmoRDataset


with open('cfgs/diffusion.yml', "r") as f:
    diff_config = yaml.safe_load(f)

new_config = dict2namespace(diff_config)

len_snippet = 32
# data_type = 'dhf1k'
data_type = 'memor'
gt_length=1
img_size = (224, 384)

config = dict(
    type=VideoSaliencyModel,
    channel_list=encoder_channel_list,
    visual_net=dict(
        type=MViT,
        arch="small",
        pretrained="/home/leessu/code/diff_sal/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth", #수정 필!!!!!!!!!!!!!!
        out_scales=[0, 1, 2, 3]),
    spatiotemp_net=None,
    decoder_net=dict(
        type=SalUNet,
        image_based=True,
        img_size=img_size,
        frames_len=gt_length,
        tasks=['futr'],
        in_index=[0, 1, 2, 3],
        idx_to_planes={
            0: 96,
            1: 192,
            2: 384,
            3: 768
        },
        mid_num_stages=4,
        temporal_size=9,
        temporal_list= [5, 5, 5, 5],
        keep_max_len=5,
        exclude_layers=[],
        futr_num_stages=0,
        ori_embed_dim=768,
        down_embed_dim=96,
        patch_size=[0, 3, 3, 3],
        patch_stride=[0, 1, 1, 1],
        patch_padding=[0, 2, 2, 2],
        up_channel=[768, 384, 192, 96],
        num_heads=[2, 2, 2, 2],
        mlp_ratio=[2., 2., 2., 2.],
        drop_path_rate=[0.15, 0.15, 0.15, 0.15],
        qkv_bias=[True, True, True, True],
        kv_proj_method=['avg', 'avg', 'avg', 'avg'],
        kernel_kv=[2, 4, 8, 16],
        padding_kv=[0, 0, 0, 0],
        stride_kv=[2, 4, 8, 16],
        q_proj_method=['dw_bn', 'dw_bn', 'dw_bn', 'dw_bn'],
        kernel_q=[3, 3, 3, 3],
        padding_q=[1, 1, 1, 1],
        stride_q=[1, 1, 1, 1],
    ))

data_dict = { #수정 필요
    'dhf1k': {
        'type': DHF1KDatasetMultiFrames,
        'path': 'VideoSalPrediction/DHF1k_extracted',
    },
    'holly': {
        'type': HollyDataset,
        'path': 'VideoSalPrediction/Hollywood2',
    },
    'ucf': {
        'type': UCFDataset,
        'path': 'VideoSalPrediction/ucf',
    },
    'memor': {
        'type': MEmoRDataset,
        'path': 'VideoSalPrediction/memor',
    }
}

data = dict(train=dict(
    type=data_dict[data_type]['type'],
    path_data=data_dict[data_type]['path'],
    len_snippet=len_snippet,
    mode="train",
    img_size=img_size,
    alternate=1,
    gt_length=gt_length,
    ),
    val=dict(
        type=data_dict[data_type]['type'],
        path_data=data_dict[data_type]['path'],
        len_snippet=len_snippet,
        mode="test",
        img_size=img_size,
        alternate=1,
        gt_length=gt_length,
    )
)

