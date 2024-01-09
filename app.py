"""
@project:model_deployment
@Author: Phantom
@Time:2024/1/2 下午1:44
@Email: 2909981736@qq.com
"""

import os
import json
import time
import shutil
import uvicorn
import zipfile
import threading
import subprocess
from utils.get_resource_occupation import get_resource_occupation
from utils.UUID import UUID
from fastapi import UploadFile
from collections import OrderedDict
from config.app_config import init_config
from fastapi.responses import JSONResponse
from utils.model_handler import ModelHandler
from utils.cleanup import cleanup_cuda_cache
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Form
from utils.check_installed_package import check_package_installed

model_instances = OrderedDict()
model_cleanup_interval = init_config.model_cleanup_interval
tempdir_cleanup_interval = init_config.tempdir_cleanup_interval
model_repo = init_config.model_repo
model_temp = init_config.model_temp
max_model_count = init_config.max_model_count
model_count = len([name for name in os.listdir(model_repo) if
                   os.path.isdir(os.path.join(model_repo, name)) and name != "__pycache__"])

app = FastAPI()
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有标头
)


@app.get("/")
async def root():
    return {"message": "Hello AI World!"}


def create_dynamic_route(model_repo: str, model_dir: str):
    model_instance = ModelHandler(model_repo, model_dir)

    # 创建predict的路由
    @app.post(f"/predict/{model_repo}/{model_dir}")
    async def make_prediction_dynamic(input_data: dict):
        try:
            if model_instance not in model_instances:
                model_instances[model_instance] = time.time()
            else:
                model_instances.move_to_end(model_instance)
            result = model_instance.predict(input_data)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def update_dynamic_routes(model_repo: str):
    for model_dir in os.listdir(model_repo):
        model_dir_path = os.path.join(model_repo, model_dir)
        if os.path.isdir(model_dir_path):
            # 检查是否已经存在对应的路由，避免重复创建
            predict_route_exists = any(
                route.path == f"/predict/{model_repo}/{model_dir}" and "POST" in route.methods for route in
                app.router.routes)
            if not predict_route_exists:
                create_dynamic_route(model_repo, model_dir)


@app.get("/select_model")
async def select_model(request: Request):
    try:
        host = request.headers.get("Host")
        model_route = []
        for route in [route for route in app.router.routes if f"/predict/{model_repo}/" in route.path]:
            model_predict_api = f"http://{host}{route.path}"
            with open(os.path.join(route.path.replace("/predict/", ""), 'config.json')) as f:
                model_config = json.load(f)
            model_route.append({"model_predict_api": model_predict_api, "model_config": model_config})

        return JSONResponse(content=model_route)
    except Exception as e:
        raise HTTPException(status_code=511, detail=str(e))


@app.get("/select_model_by_uuid")
async def select_model_by_uuid(model_uuid: str, request: Request):
    try:
        host = request.headers.get("Host")
        for route in [route for route in app.router.routes if f"/predict/{model_repo}/{model_uuid}" in route.path]:
            model_predict_api = f"http://{host}{route.path}"
            with open(os.path.join(route.path.replace("/predict/", ""), 'config.json')) as f:
                model_config = json.load(f)
            return JSONResponse(content={"model_predict_api": model_predict_api, "model_config": model_config})
        return JSONResponse(content={"model_predict_api": None, "model_config": None})
    except Exception as e:
        raise HTTPException(status_code=511, detail=str(e))


@app.post("/upload_model")
async def upload_model(file: UploadFile = Form(...)):
    global model_count
    try:
        if model_count + 1 > max_model_count:
            raise HTTPException(status_code=520, detail="系统模型数量超过上限，上传失败")
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=512, detail="仅支持上传ZIP压缩包。")
        model_temp_path = os.path.join(model_temp, file.filename)
        # 保存上传的压缩包到临时位置
        with open(model_temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # 打开压缩包检查是否含有config.json文件（意在检查是否是一级目录，压缩包要保证是对文件直接进行的压缩，不含文件夹）
        with zipfile.ZipFile(model_temp_path, "r") as zfile:
            if "config.json" not in zfile.namelist():
                os.remove(model_temp_path)
                raise HTTPException(status_code=512,
                                    detail="压缩包中缺少 config.json 文件。提示：请直接对所需文件进行压缩，不要对文件夹进行压缩")

        # 生成新上传模型的uuid文件夹
        model_dir = UUID.get_timestamp_uuid()
        model_path = os.path.join(model_repo, model_dir)
        while os.path.exists(model_path):
            model_dir = UUID.get_timestamp_uuid()
            model_path = os.path.join(model_repo, model_dir)
        os.mkdir(model_path)

        # 解压文件夹压缩包
        shutil.unpack_archive(model_temp_path, model_path)
        os.remove(model_temp_path)

        # 动态添加路由
        create_dynamic_route(model_repo, model_dir)

        model_count += 1
        return JSONResponse(content={"message": "模型上传成功", "model_uuid": model_dir})
    except Exception as e:
        raise HTTPException(status_code=512, detail=str(e))


@app.get("/delete_model")
async def deleted_model(model_uuid: str):
    folder_path = os.path.join(model_repo, model_uuid)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 删除文件夹
        try:
            shutil.rmtree(folder_path)
            print(folder_path, "is deleted")
            app.router.routes = [route for route in app.router.routes if route.path != f"/predict/{folder_path}"]
            return {"message": f"Folder '{model_uuid}' deleted successfully"}
        except OSError as e:
            raise HTTPException(status_code=513, detail=f"Error deleting model '{model_uuid}': {str(e)}")
    else:
        return HTTPException(status_code=513, detail=f"model '{model_uuid}' does not exist")


@app.get("/install_package")
async def install_package(package: str, version: str, mirror: str = None):
    package = package.strip()
    if "=" in package:
        package = package.split("=")[0]
    is_installed = check_package_installed(package)
    if is_installed:
        return HTTPException(status_code=514, detail=f"安装失败，({package})已经被安装了，版本：{is_installed}")
    try:
        package = f"{package}=={version}"
        if isinstance(mirror, str):
            mirror = mirror.strip()
        if mirror is None or mirror == "":
            subprocess.check_call(["pip", "install", package])
        else:
            subprocess.check_call(["pip", "install", "--index-url", mirror, package])
        return HTTPException(status_code=200, detail=f"安装成功：{package}")
    except subprocess.CalledProcessError as e:
        return HTTPException(status_code=514, detail=f"安装失败，安装库时发生错误: {str(e)}")


@app.post("/upload_predict_data_file")
async def upload_predict_data_file(file: UploadFile = Form(...)):
    try:
        # 生成上传测试数据的uuid文件夹
        predict_data_path = os.path.join(model_temp, UUID.get_timestamp_uuid())
        while os.path.exists(predict_data_path):
            predict_data_path = os.path.join(model_temp, UUID.get_timestamp_uuid())
        os.mkdir(predict_data_path)
        predict_data_file_path = os.path.join(predict_data_path, file.filename)
        with open(predict_data_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return JSONResponse(content={"message": "文件上传成功", "file_path": predict_data_file_path})
    except Exception as e:
        return HTTPException(status_code=515, detail=f"上传测试数据文件失败，错误原因：{str(e)}")


@app.get("/get_server_resource_occupation")
async def get_server_resource_occupation():
    try:
        cpu_usage, gpu_memory_usage, memory_usage, disk_usage = get_resource_occupation(os.getcwd())
        return JSONResponse(
            content={"message": "服务器资源占用查询成功", "cpu_usage": cpu_usage, "gpu_memory_usage": gpu_memory_usage,
                     "memory_usage": memory_usage, "disk_usage": disk_usage})
    except Exception as e:
        return HTTPException(status_code=516, detail=f"服务器资源占用查询失败，错误原因：{str(e)}")


def cleanup_model():
    while True:
        time.sleep(model_cleanup_interval)  # 每隔一小时处理一次模型实例
        if len(model_instances) > 0:
            lru_model = next(iter(model_instances))
            model_instances.move_to_end(lru_model)
            lru_model.model = None
            lru_model.model_initialized = False
            print("lru_model", lru_model)
            cleanup_cuda_cache()


def cleanup_temp():
    while True:
        time.sleep(tempdir_cleanup_interval)  # 每隔一小时处理一次模型实例
        current_time = time.time()
        files = os.listdir(model_temp)
        for file in files:
            file_path = os.path.join(model_temp, file)
            file_create_time = os.path.getctime(file_path)
            time_difference = current_time - file_create_time
            if time_difference > tempdir_cleanup_interval:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)


# 初始加载已有的模型路由
update_dynamic_routes(model_repo)

cleanup_model_thread = threading.Thread(target=cleanup_model)
cleanup_model_thread.start()
cleanup_tempdir_thread = threading.Thread(target=cleanup_temp)
cleanup_tempdir_thread.start()

if __name__ == '__main__':
    uvicorn.run(app, port=8008)
