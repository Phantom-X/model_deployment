"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/3 下午4:44
@Email: 2909981736@qq.com
"""
import uuid


class UUID:
    def __init__(self):
        pass

    @staticmethod
    def get_timestamp_uuid():
        """根据时间戳生成 UUID，保证全球唯一"""
        uuid1 = uuid.uuid1()
        return str(uuid1).replace("-", "")

    @staticmethod
    def get_randomnumber_uuid():
        """根据随机数生成 UUID，几率超小，但使用方便"""
        uuid4 = uuid.uuid4()
        return str(uuid4).replace("-", "")

    @staticmethod
    def get_specifiedstr_uuid(name, namespace=uuid.NAMESPACE_DNS):
        """根据指定字符串生成 UUID"""
        uuid3 = uuid.uuid3(namespace, name)
        return str(uuid3).replace("-", "")

    @staticmethod
    def get_specifiedstr_SHA1_uuid(name, namespace=uuid.NAMESPACE_DNS):
        """根据指定字符串生成 UUID，使用 SHA1 散列"""
        uuid5 = uuid.uuid5(namespace, name)
        return str(uuid5).replace("-", "")

