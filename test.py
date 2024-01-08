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


def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, ".")
    print(module_path)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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
