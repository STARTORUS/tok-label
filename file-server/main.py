from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import Optional
from pydantic import BaseModel
from util import export_postgres_data
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "/data"

URL_PREFIX = "/files"
# 挂载静态文件目录，使其可通过 URL 访问
app.mount(URL_PREFIX, StaticFiles(directory=DATA_DIR, html=True), name="files")

# @app.get("/files/{file_path:path}")
# def get_file(file_path: str):
#     file_location = os.path.join(DATA_DIR, file_path)
#     return FileResponse(file_location)

# 列出文件
@app.get("/list/")
async def list_files(dir: str = None, recursive: bool = False):
    """
    列出 DATA_DIR (/data) 下的文件。

    参数
    ----
    dir        : 相对目录；None = /data 根目录
    recursive  : True 时递归列出子目录文件
    """
    base_path = os.path.join(DATA_DIR, dir) if dir else DATA_DIR

    if not os.path.exists(base_path):
        return {"error": f"Directory {dir or '/'} not found"}

    files_rel: list[str] = []
    # 是否递归查找
    if recursive:
        # os.walk 返回 (root, dirs, files)
        for root, _, filenames in os.walk(base_path):
            rel_root = os.path.relpath(root, DATA_DIR)  # 相对 DATA_DIR 的root
            for fname in filenames:
                # 拼出相对 DATA_DIR 的路径
                rel_path = os.path.join(rel_root, fname) if rel_root != "." else fname
                files_rel.append(rel_path)
    else:
        files_rel = os.listdir(base_path)

    files_rel.sort()

    file_urls = [f"{URL_PREFIX}/{path}" for path in files_rel]
    return {"files": files_rel, "urls": file_urls}

# 删除文件
@app.delete("/delete/")
async def delete_file(dir: str, file: Optional[str]=None):
    target = os.path.join(DATA_DIR, dir)
    # 检查目录是否存在
    if not os.path.exists(target):
        return {"error": f"Directory {dir} not found"}
    if file is None:
        # 删除目录及目录下的所有文件
        shutil.rmtree(target)
        return {"message": f"Directory {dir} and all contents deleted successfully"}
    # 检查文件是否存在
    if not os.path.exists(os.path.join(target, file)):
        return {"error": f"File {file} not found"}
    # 删除文件
    os.remove(os.path.join(DATA_DIR, dir, file))
    return {"message": f"File {file} deleted successfully"}

# upload file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 保存文件
    file_path = os.path.join(DATA_DIR, file.filename)
    # 如果文件已存在，则删除
    if os.path.exists(file_path):
        os.remove(file_path)
    # 如果目录不存在，则创建
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {
        "message": f"File {file.filename} uploaded successfully", 
        "url": f"{URL_PREFIX}/{file.filename}"
        }

# 导出数据
class ExportRequest(BaseModel):
    project_name: str
    shots: int|list[int]
    name_table_columns: dict
    t_min: float = -np.inf
    t_max: float = np.inf
    resolution: float = 1e-3

# convert pgdata to csv
@app.post("/export/")
async def export_pgdata(request: ExportRequest):
    if isinstance(request.shots, int):
        request.shots = [request.shots]
    # get data from postgres
    data = export_postgres_data(request.shots, request.name_table_columns, request.t_min, request.t_max, request.resolution)
    # save data to csv
    os.makedirs(os.path.join(DATA_DIR, request.project_name), exist_ok=True)
    for shot in request.shots:
        file_name = f"{request.project_name}/{shot}.csv"
        data[shot].to_csv(os.path.join(DATA_DIR, file_name), index=False, encoding='utf-8')
    # return url
    return {
        "message": "Data exported successfully", 
        "urls": [f"{URL_PREFIX}/{request.project_name}/{shot}.csv" for shot in request.shots]
        }
