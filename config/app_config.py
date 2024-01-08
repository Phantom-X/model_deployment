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
    model_cleanup_interval: int = 20  # 每隔一小时处理一次模型实例


@lru_cache
def get_init_config():
    return Init_Config()


init_config = get_init_config()
