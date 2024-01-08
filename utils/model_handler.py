"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/2 下午4:47
@Email: 2909981736@qq.com
"""
import os
import json
from utils.dynamic_load import get_class_in_module
import torch


class ModelHandler:
    def __init__(self, model_repo, model_dir):
        self.model = None
        self.model_initialized = False
        self.model_repo = model_repo
        self.model_dir = model_dir
        with open(os.path.join(model_repo, self.model_dir, 'config.json')) as f:
            self.config = json.load(f)
        self.model_name = self.config["model_name"]
        self.preprocess = get_class_in_module(self.config["preprocess_function"],
                                              os.path.join(model_repo, self.model_dir,
                                                           self.config["preprocess_function"]))
        self.postprocess = get_class_in_module(self.config["postprocess_function"],
                                               os.path.join(model_repo, self.model_dir,
                                                            self.config["postprocess_function"]))

    def initialize_model(self):
        Model = get_class_in_module(self.model_name, os.path.join(self.model_repo, self.model_dir, self.model_name))
        model = Model(**self.config["model_params"])
        model.load_state_dict(torch.load(os.path.join(self.model_repo, self.model_dir, self.config["model_weights"])))
        device = self.config["gpu"] if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        print(f"Initializing model: {self.model_name}")
        self.model_initialized = True

    def predict(self, input_data):
        if not self.model_initialized:
            self.initialize_model()
        self.model.eval()
        tensor = self.preprocess(input_data, self.config)
        with torch.no_grad():
            output = self.model(tensor)

        resp = self.postprocess(output, self.config)

        print(f"Predicting with model: {self.model_name}")
        return {"model_name": self.model_name, "prediction": resp}

    def train(self):
        pass
