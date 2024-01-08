"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/4 下午6:18
@Email: 2909981736@qq.com
"""
import subprocess


def check_package_installed(package_name):
    output = subprocess.check_output(["pip", "list"])
    packages = output.decode("utf-8").strip().split("\n")[2:]
    for package in packages:
        package_data = package.split()
        if package_name == package_data[0]:
            return package_data[1]
    return None

