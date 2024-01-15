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

# 在本地测试自己的模型代码包
input_model_path = 'model_repo/e6ae239aaf7111ee8623cf3bf55179fe'  # 填写模型代码包本地路径
input_data = {"text": "财联社1月15日电，今日有2只新股申购，分别为创业板的美信科技，沪市主板的盛景微；无新股上市。"}


def get_class_in_module(class_name, module_path):
    module_path = module_path.replace(os.path.sep, ".")
    print(module_path)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    with open(os.path.join(input_model_path, 'config.json')) as f:
        config = json.load(f)
    print(config)
    Model = get_class_in_module(config["model_name"], os.path.join(input_model_path, config["model_name"]))
    model = load_weights(Model, config)
    preprocess = get_class_in_module(config["preprocess_function"],
                                     os.path.join(input_model_path, config["preprocess_function"]))
    postprocess = get_class_in_module(config["postprocess_function"],
                                      os.path.join(input_model_path, config["postprocess_function"]))
    tensor = preprocess(input_data, config)
    with torch.no_grad():
        if isinstance(tensor, dict):
            output = model(**tensor)
        else:
            output = model(tensor)
    resp = postprocess(output, config)
    print(resp)


def load_weights(Model, config):
    if config["weight_load_method"] == "" or config["weight_load_method"] == "default":
        model = Model(**config["model_params"])
        model.load_state_dict(torch.load(os.path.join(input_model_path, config["model_weights"])))
        model.eval()
    elif config["weight_load_method"] == "jit":
        model = torch.jit.load(os.path.join(input_model_path, config["model_weights"]))
    elif config["weight_load_method"] == "ultralytics":
        model = Model(os.path.join(input_model_path, config["model_weights"]))
    elif config["weight_load_method"] == "transformers":
        raise Exception(f"暂未开通，请等待后续平台升级")
    else:
        raise Exception(f"没有这种模型权重导入方式")
    device = config["gpu"] if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


if __name__ == '__main__':
    main()
