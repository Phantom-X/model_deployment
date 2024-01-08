"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/5 下午7:22
@Email: 2909981736@qq.com
"""
import torch


def cleanup_model_cuda_cache():
    torch.cuda.empty_cache()
