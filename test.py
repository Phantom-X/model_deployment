"""
@project:model_deployment
@Author: Phantom
@Time:2023/12/29 下午7:48
@Email: 2909981736@qq.com
"""
import os
import importlib
import json
import torch
import psutil
import GPUtil


def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, ".")
    print(module_path)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    input_model_path = 'model_repo/textcnn'

    with open(os.path.join(input_model_path, 'config.json')) as f:
        config = json.load(f)

    # print(config)

    Model = get_class_in_module(config["model_name"], os.path.join(input_model_path, config["model_name"]))

    model = Model(**config["model_params"])
    model.load_state_dict(torch.load(config["model_weights"]))
    device = config["gpu"] if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    preprocess = get_class_in_module(config["preprocess_function"],
                                     os.path.join(input_model_path, config["preprocess_function"]))

    input_text = "教育教育教育教育教育教育教育教育"
    tensor = preprocess(input_text, config)
    with torch.no_grad():
        output = model(tensor)
        print(output)

    postprocess = get_class_in_module(config["postprocess_function"],
                                      os.path.join(input_model_path, config["postprocess_function"]))

    resp = postprocess(output, config)

    print(resp)


if __name__ == '__main__':
    # main()
    # 获取CPU利用率
    cpu_usage = psutil.cpu_percent()
    # 获取内存占用率
    memory_usage = psutil.virtual_memory().percent
    # 获取磁盘使用情况
    disk_usage = psutil.disk_usage(os.getcwd())
    # 获取所有可用的GPU
    gpus = GPUtil.getGPUs()
    # 遍历每个GPU并获取显存占用率
    for gpu in gpus:
        memory_usage = gpu.memoryUtil * 100
        print(f"显卡显存占用率: {memory_usage}%")

    print(f"CPU利用率: {cpu_usage}%")
    print(f"内存占用率: {memory_usage}%")
    print(f"磁盘使用情况: {disk_usage}")
