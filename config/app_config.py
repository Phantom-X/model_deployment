"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/5 下午1:12
@Email: 2909981736@qq.com
"""
from pydantic.v1 import BaseSettings
from functools import lru_cache


class Init_Config(BaseSettings):
    model_repo: str = "model_repo"
    model_temp: str = "temp"
    max_model_count: int = 520
    model_cleanup_interval: int = 30  # 每隔30s处理一次模型实例
    tempdir_cleanup_interval: int = 1800  # 每隔0.5小时处理一次temp文件夹
    eureka_registration_server: str = "http://100.100.30.52:7895"  # http://100.100.30.52:7895
    app_name: str = "model-deployment-app"
    instance_port: int = 8008
    instance_host: str = "0.0.0.0"


@lru_cache
def get_init_config():
    return Init_Config()


init_config = get_init_config()
