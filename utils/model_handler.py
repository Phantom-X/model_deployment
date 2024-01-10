"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/2 下午4:47
@Email: 2909981736@qq.com
"""
import os
import json
import torch
from utils.cleanup import cleanup_cuda_cache
from utils.dynamic_load import get_class_in_module


class ModelHandler:
    def __init__(self, model_repo, model_dir):
        self.model = None
        self.model_initialized = False
        self.model_repo = model_repo
        self.model_dir = model_dir
        with open(os.path.join(model_repo, self.model_dir, 'config.json')) as f:
            self.config = json.load(f)
        self.model_name = self.config["model_name"]
        self.weight_load_method = self.config["weight_load_method"]
        self.preprocess = get_class_in_module(self.config["preprocess_function"],
                                              os.path.join(model_repo, self.model_dir,
                                                           self.config["preprocess_function"]))
        self.postprocess = get_class_in_module(self.config["postprocess_function"],
                                               os.path.join(model_repo, self.model_dir,
                                                            self.config["postprocess_function"]))

    def initialize_model(self):
        Model = get_class_in_module(self.model_name, os.path.join(self.model_repo, self.model_dir, self.model_name))
        try:
            if self.weight_load_method == "" or self.weight_load_method == "default":
                model = Model(**self.config["model_params"])
                model.load_state_dict(
                    torch.load(os.path.join(self.model_repo, self.model_dir, self.config["model_weights"])))
                model.eval()
            elif self.weight_load_method == "jit":
                model = torch.jit.load(os.path.join(self.model_repo, self.model_dir, self.config["model_weights"]))
            elif self.weight_load_method == "ultralytics":
                model = Model(os.path.join(self.model_repo, self.model_dir, self.config["model_weights"]))
            elif self.weight_load_method == "transformers":
                raise Exception(f"暂未开通，请等待后续平台升级，weight_load_method={self.weight_load_method}")
            else:
                raise Exception(f"没有这种模型权重导入方式，weight_load_method={self.weight_load_method}")
        except Exception as e:
            raise Exception(f"模型权重导入失败: {e}")

        device = self.config["gpu"] if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        print(f"Initializing model: {self.model_name}")
        self.model_initialized = True

    def predict(self, input_data):
        if not self.model_initialized:
            self.initialize_model()
        tensor = self.preprocess(input_data, self.config)
        with torch.no_grad():
            if isinstance(tensor, dict):
                output = self.model(**tensor)
            else:
                output = self.model(tensor)
        resp = self.postprocess(output, self.config)
        cleanup_cuda_cache()

        print(f"Predicting with model: {self.model_name}")
        return {"predict": resp}

    def eval(self):
        pass

    def train(self):
        pass
