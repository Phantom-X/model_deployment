"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/9 下午7:44
@Email: 2909981736@qq.com
"""
import psutil
import GPUtil


def get_resource_occupation(path):
    # 获取CPU利用率
    cpu_usage = psutil.cpu_percent()
    # 获取内存占用率
    memory_usage = psutil.virtual_memory().percent
    # 获取磁盘使用情况
    disk_usage = psutil.disk_usage(path)
    # 获取所有可用的GPU
    gpus = GPUtil.getGPUs()
    # 遍历每个GPU并获取显存占用率
    gpu_memory_usage = []
    for gpu in gpus:
        gpu_memory_usage.append(gpu.memoryUtil * 100)

    return cpu_usage, gpu_memory_usage, memory_usage, disk_usage
